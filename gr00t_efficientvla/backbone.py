# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3 backbone with EfficientVLA layer pruning and in-LLM visual token pruning.

The decoder loop of ``Qwen3VLTextModel.forward`` (transformers 4.57) is written out so that:

- layers in ``drop_layers`` are skipped (identity); DeepStack features are still added at
  their original layer index, so dropping one of the first layers keeps the injection;
- after the first ``prune_after`` layers, image tokens are scored with the attention of the
  last of them, selected per view (key / task / diversity), and the hidden states, masks,
  MRoPE ids and the remaining DeepStack features are gathered to the kept tokens.

Input preparation (embeddings, ViT, MRoPE ids) and text / view bookkeeping are shared with
``gr00t_adp.Qwen3BackboneADP``, run with its own pruning disabled.
"""

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
from gr00t_adp.config import ADPConfig
from gr00t_adp.qwen3_backbone_adp import Qwen3BackboneADP
from gr00t_adp.token_scoring import allocate_budget
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.masking_utils import create_causal_mask

from .config import EfficientVLAConfig
from .token_selection import select_key_task_diverse, text_to_image_attention


class Qwen3BackboneEfficientVLA(Qwen3BackboneADP):
    evla_config: EfficientVLAConfig
    evla_last: dict | None

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        prepared = self.prepare_pruned_inputs(vl_input)
        hidden, state = self.run_layers(prepared)
        return BatchFeature(
            data={
                "backbone_features": hidden,
                "backbone_attention_mask": state["attention_mask"] == 1,
                "image_mask": state["image_mask"],
            }
        )  # [B, T2, hidden_size]

    def run_layers(self, prepared: dict, emulate_drop: bool = False):
        """Decoder loop. Returns the last kept layer's output (pre-norm) and the final state.

        With ``emulate_drop`` the pruned tokens stay in the sequence and are only masked out as
        keys from layer ``prune_after`` on; tests use it as the full-length reference.
        """
        cfg = self.evla_config
        lm = self.model.model.language_model
        n_layers = len(lm.layers)
        prune_at = cfg.prune_after if cfg.keep_ratio < 1.0 else None
        if prune_at is not None and prune_at >= n_layers:
            raise ValueError(f"prune_after={prune_at} must be < number of layers {n_layers}")

        state = {
            "hidden": prepared["inputs_embeds"],
            "attention_mask": prepared["attention_mask"],
            "position_ids": prepared["position_ids"],
            "image_mask": prepared["image_mask"],
            "deepstack": list(prepared["deepstack_embeds"]),
            "input_ids": prepared["input_ids"],
        }
        self._refresh_layer_inputs(lm, state)
        self.evla_last = None
        scores = None
        for idx, layer in enumerate(lm.layers):
            if idx == prune_at:
                keep, record = self._select_tokens(state, scores)
                self.evla_last = record if cfg.record else None
                state["keep"] = keep
                if emulate_drop:
                    state["attention_mask"] = state["attention_mask"] * keep.to(
                        state["attention_mask"].dtype
                    )
                    self._refresh_layer_inputs(lm, state)
                else:
                    self._gather(lm, state, keep, first_deepstack=idx)
            if prune_at is not None and idx == prune_at - 1:
                scores = self._relevance(layer, state)
            if idx not in cfg.drop_layers:
                state["hidden"] = layer(
                    state["hidden"],
                    attention_mask=state["mask"],
                    position_ids=state["position_ids"][0],
                    past_key_values=None,
                    cache_position=state["cache_position"],
                    position_embeddings=state["position_embeddings"],
                )
            if idx < len(state["deepstack"]):
                state["hidden"] = lm._deepstack_process(
                    state["hidden"], state["image_mask"], state["deepstack"][idx]
                )
        return state["hidden"], state

    def _refresh_layer_inputs(self, lm, state):
        hidden = state["hidden"]
        cache_position = torch.arange(hidden.shape[1], device=hidden.device)
        state["cache_position"] = cache_position
        state["mask"] = create_causal_mask(
            config=lm.config,
            input_embeds=hidden,
            attention_mask=state["attention_mask"],
            cache_position=cache_position,
            past_key_values=None,
            position_ids=state["position_ids"][0],
        )
        state["position_embeddings"] = lm.rotary_emb(hidden, state["position_ids"])

    def _gather(self, lm, state, keep, first_deepstack):
        n_keep = keep.sum(dim=1)
        assert torch.all(n_keep == n_keep[0]), "every sample must keep the same token count"
        idx = keep.nonzero()[:, 1].view(keep.shape[0], -1)  # (B, L'), ascending
        hidden = state["hidden"]
        # DeepStack features are flattened in (batch, position) order of the image tokens.
        flat_keep = keep[state["image_mask"]]
        state["deepstack"] = [
            d[flat_keep] if i >= first_deepstack else d for i, d in enumerate(state["deepstack"])
        ]
        state["hidden"] = torch.gather(hidden, 1, idx[..., None].expand(-1, -1, hidden.shape[-1]))
        state["attention_mask"] = torch.gather(state["attention_mask"], 1, idx)
        state["position_ids"] = torch.gather(state["position_ids"], 2, idx[None].expand(3, -1, -1))
        state["image_mask"] = torch.gather(state["image_mask"], 1, idx)
        state["input_ids"] = torch.gather(state["input_ids"], 1, idx)
        self._refresh_layer_inputs(lm, state)

    @torch.no_grad()
    def _relevance(self, layer, state) -> list[torch.Tensor]:
        """Per-sample raw relevance of every image token from ``layer``'s attention."""
        cos, sin = state["position_embeddings"]
        out = []
        for b in range(state["hidden"].shape[0]):
            ids, mask = state["input_ids"][b], state["attention_mask"][b]
            image_mask = state["image_mask"][b]
            image_pos = image_mask.nonzero().squeeze(-1)
            text_pos = self._text_positions(ids, mask, image_mask)
            out.append(
                text_to_image_attention(
                    layer, state["hidden"][b], cos[b], sin[b], mask == 1, text_pos, image_pos
                )
            )
        return out

    @torch.no_grad()
    def _select_tokens(self, state, scores):
        """Keep mask (B, L) over the current sequence, selecting per view."""
        cfg = self.evla_config
        counts = self._image_counts(state["image_mask"].shape[0])
        budget = allocate_budget(counts, cfg.keep_ratio, "per_view_uniform")
        views = self._image_view_names(len(counts))
        keep = torch.ones_like(state["attention_mask"], dtype=torch.bool)
        record = {"counts": counts, "views": views, "grid_thw": self._grid_thw, "selections": []}
        for b, sample_scores in enumerate(scores):
            image_pos = state["image_mask"][b].nonzero().squeeze(-1)
            features = state["hidden"][b, image_pos]
            drop = torch.ones(image_pos.numel(), dtype=torch.bool, device=image_pos.device)
            selections = []
            offset = 0
            for n, k in zip(counts, budget):
                sel = select_key_task_diverse(
                    sample_scores[offset : offset + n],
                    features[offset : offset + n],
                    k,
                    cfg.num_key_tokens,
                    cfg.alpha,
                )
                drop[sel["kept"] + offset] = False
                selections.append({name: t.cpu() for name, t in sel.items()})
                offset += n
            keep[b, image_pos[drop]] = False
            record["selections"].append(selections)
        return keep, record

    def _image_counts(self, batch_size: int) -> list[int]:
        merge = self.model.model.visual.spatial_merge_size
        counts_all = (self._grid_thw_all.prod(-1) // merge**2).tolist()
        assert len(counts_all) % batch_size == 0, "samples must have the same number of images"
        counts = counts_all[: len(counts_all) // batch_size]
        assert counts_all == counts * batch_size, "samples must have identical image layouts"
        return counts

    def prepare_pruned_inputs(self, vl_input: BatchFeature, apply_drop: bool = True) -> dict:
        prepared = super().prepare_pruned_inputs(vl_input, apply_drop=False)
        prepared["input_ids"] = vl_input["input_ids"]
        self._grid_thw_all = vl_input["image_grid_thw"]
        n_per_sample = self._grid_thw_all.shape[0] // vl_input["input_ids"].shape[0]
        self._grid_thw = self._grid_thw_all[:n_per_sample].tolist()
        return prepared


def enable_efficientvla_backbone(
    backbone: Qwen3Backbone, config: EfficientVLAConfig
) -> Qwen3BackboneEfficientVLA:
    """Switch a loaded ``Qwen3Backbone`` to the EfficientVLA forward in place."""
    if not isinstance(backbone, Qwen3Backbone):
        raise TypeError(f"expected Qwen3Backbone, got {type(backbone).__name__}")
    n_layers = len(backbone.model.model.language_model.layers)
    bad = [i for i in config.drop_layers if not 0 <= i < n_layers]
    if bad:
        raise ValueError(f"drop_layers {bad} out of range for {n_layers} layers")
    backbone.__class__ = Qwen3BackboneEfficientVLA
    backbone.evla_config = config
    backbone.evla_last = None
    # Bookkeeping helpers inherited from the ADP backbone read this; its pruning stays off.
    backbone.adp_config = ADPConfig(
        keep_ratio=1.0, text_source=config.text_source, view_keys=config.view_keys
    )
    backbone.adp_last = None
    return backbone


def disable_efficientvla_backbone(backbone: Qwen3BackboneEfficientVLA) -> Qwen3Backbone:
    backbone.__class__ = Qwen3Backbone
    return backbone
