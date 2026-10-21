# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3 backbone with VLA-Pruner (arXiv 2511.16449) for GR00T N1.7, batch size 1.

Per inference, as in the reference OpenVLA-OFT code:

1. LLM layers 0..K run on every token. Semantic relevance = attention of layer K, averaged
   over heads and over all (valid) query rows, on each image token (paper Eq. 2).
2. Action relevance = decaying window average (w=3, gamma=0.8) of the DiT action -> image
   attention of the previous inferences (``ActionAttentionRecorder``); pruned tokens score 0.
   Until w inferences are recorded, nothing is pruned.
3. Per view: top-k by semantic union top-k by action; if larger than k, greedy max-min cosine
   selection on the tokens' LLM input embeddings (Combine-then-Filter).
4. The kept tokens continue through layers K+1.. (the in-LLM gather of the EfficientVLA backbone:
   hidden states, MRoPE ids, masks, DeepStack features).
"""

from collections import deque

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
from gr00t_adp.token_scoring import allocate_budget
from gr00t_efficientvla.backbone import Qwen3BackboneEfficientVLA, enable_efficientvla_backbone
from gr00t_efficientvla.config import EfficientVLAConfig
from gr00t_efficientvla.token_selection import text_to_image_attention
import torch
from transformers.feature_extraction_utils import BatchFeature

from .action_attention import ActionAttentionRecorder
from .config import VLAPrunerConfig
from .selection import combine_then_filter, smoothed_history


class Qwen3BackboneVLAPruner(Qwen3BackboneEfficientVLA):
    vlap_config: VLAPrunerConfig
    vlap_recorder: ActionAttentionRecorder | None
    vlap_history: deque
    vlap_last: dict | None

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        if vl_input["input_ids"].shape[0] != 1:
            raise ValueError("VLA-Pruner supports batch size 1 only")
        cfg = self.vlap_config
        prepared = self.prepare_pruned_inputs(vl_input)
        image_mask_full = prepared["image_mask"][0]
        n_image = int(image_mask_full.sum())

        self._flush_action_scores(n_image)
        historical = smoothed_history(self.vlap_history, cfg.history_decay) if cfg.enabled else None
        self._vlap_action = historical
        self._vlap_features = prepared["inputs_embeds"][0, image_mask_full].detach().clone()
        prune = cfg.enabled and historical is not None and cfg.keep_ratio < 1.0
        self.evla_config = EfficientVLAConfig(
            keep_ratio=cfg.keep_ratio if prune else 1.0,
            prune_after=cfg.prune_layer + 1,
            cache_interval=1,
            view_keys=cfg.view_keys,
            record=cfg.record,
        )
        hidden, state = self.run_layers(prepared)

        # Map the image columns the DiT will see back to the full image-token index.
        keep = state.get("keep")
        kept_image = torch.arange(n_image, device=hidden.device)
        if keep is not None:
            kept_image = kept_image[keep[0][image_mask_full]]
        self._vlap_kept_image = kept_image if cfg.enabled else None
        self.vlap_last = self.evla_last if cfg.record else None
        return BatchFeature(
            data={
                "backbone_features": hidden,
                "backbone_attention_mask": state["attention_mask"] == 1,
                "image_mask": state["image_mask"],
            }
        )  # [B, T2, hidden_size]

    def reset_history(self) -> None:
        self.vlap_history.clear()
        self._vlap_kept_image = None
        if self.vlap_recorder is not None:
            self.vlap_recorder.pop()

    def _flush_action_scores(self, n_image: int) -> None:
        """Turn the DiT attention of the previous inference into a full-length history entry."""
        if self.vlap_recorder is None:
            return
        scores = self.vlap_recorder.pop()
        kept = getattr(self, "_vlap_kept_image", None)
        if scores is None or kept is None or not self.vlap_config.enabled:
            return
        image_cols = self._vlap_last_image_mask
        entry = torch.zeros(n_image, dtype=torch.float32)
        entry[kept.cpu()] = scores[0][image_cols].float().cpu()
        if self.vlap_history and self.vlap_history[-1].numel() != n_image:
            self.vlap_history.clear()  # image layout changed: start a new history
        self.vlap_history.append(entry)

    def _relevance(self, layer, state):
        """Semantic relevance: layer-K attention on image tokens, mean over all valid queries."""
        cos, sin = state["position_embeddings"]
        mask = state["attention_mask"][0]
        valid = mask == 1
        image_pos = state["image_mask"][0].nonzero().squeeze(-1)
        rows = valid.nonzero().squeeze(-1)
        scores = text_to_image_attention(
            layer, state["hidden"][0], cos[0], sin[0], valid, rows, image_pos
        )
        return [scores / rows.numel()]

    @torch.no_grad()
    def _select_tokens(self, state, scores):
        cfg = self.vlap_config
        counts = self._image_counts(1)
        budget = allocate_budget(counts, cfg.keep_ratio, "per_view_uniform")
        image_pos = state["image_mask"][0].nonzero().squeeze(-1)
        semantic = scores[0]
        action = self._vlap_action.to(semantic.device)
        features = self._vlap_features
        drop = torch.ones(image_pos.numel(), dtype=torch.bool, device=image_pos.device)
        selections, offset = [], 0
        for n, k in zip(counts, budget):
            view = slice(offset, offset + n)
            sel = combine_then_filter(semantic[view], action[view], features[view], k)
            drop[sel["kept"] + offset] = False
            selections.append({name: t.cpu() for name, t in sel.items()})
            offset += n
        keep = torch.ones_like(state["attention_mask"], dtype=torch.bool)
        keep[0, image_pos[drop]] = False
        record = {
            "counts": counts,
            "views": self._image_view_names(len(counts)),
            "grid_thw": self._grid_thw,
            "semantic": semantic.cpu(),
            "action": action.cpu(),
            "selections": [selections],
        }
        return keep, record


def _remember_image_mask(module, _args, output):
    module._vlap_last_image_mask = output["image_mask"][0].cpu()


def enable_vlapruner(model, config: VLAPrunerConfig) -> Qwen3BackboneVLAPruner:
    """Switch a loaded GR00T N1.7 model to VLA-Pruner in place (history emptied)."""
    backbone = model.backbone
    disable_vlapruner(model)
    enable_efficientvla_backbone(
        backbone,
        EfficientVLAConfig.disabled(view_keys=config.view_keys, text_source="all_non_image"),
    )
    n_layers = len(backbone.model.model.language_model.layers)
    if not config.prune_layer + 1 < n_layers:
        raise ValueError(f"prune_layer={config.prune_layer} leaves no layer to prune ({n_layers})")
    backbone.__class__ = Qwen3BackboneVLAPruner
    backbone.vlap_config = config
    backbone.vlap_history = deque(maxlen=config.history_window)
    backbone.vlap_last = None
    backbone._vlap_kept_image = None
    backbone.vlap_recorder = (
        ActionAttentionRecorder(model.action_head, config.action_blocks, config.action_steps)
        if config.enabled
        else None
    )
    backbone._vlap_hook = backbone.register_forward_hook(_remember_image_mask)
    return backbone


def disable_vlapruner(model) -> Qwen3Backbone:
    backbone = model.backbone
    recorder = getattr(backbone, "vlap_recorder", None)
    if recorder is not None:
        recorder.remove()
        backbone.vlap_recorder = None
    hook = getattr(backbone, "_vlap_hook", None)
    if hook is not None:
        hook.remove()
        backbone._vlap_hook = None
    backbone.__class__ = Qwen3Backbone
    return backbone
