# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3 backbone with ADP visual-token pruning before the LLM.

``Qwen3BackboneADP`` subclasses the original ``Qwen3Backbone`` and replaces only ``forward``.
The Qwen3-VL forward (``Qwen3VLModel.forward`` in transformers 4.57) is unrolled so image
tokens can be dropped after the vision encoder and before the language model:

    embed tokens -> ViT -> scatter image embeds -> 3D MRoPE ids on the full sequence
    -> score image tokens with layer-0 Q(text).K(image) -> keep top-k per image
    -> gather embeds / attention mask / MRoPE ids / DeepStack features -> language model

Dropped tokens are removed from the sequence, so the action head (vl_self_attention and
the DiT cross-attention) also only sees the kept tokens.
"""

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
import torch
from transformers.feature_extraction_utils import BatchFeature

from .config import ADPConfig
from .token_scoring import select_tokens, text_to_image_scores


VISION_END_TOKEN = "<|vision_end|>"
IM_END_TOKEN = "<|im_end|>"


class Qwen3BackboneADP(Qwen3Backbone):
    adp_config: ADPConfig
    adp_last: dict | None

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        prepared = self.prepare_pruned_inputs(vl_input)
        if self.adp_config.record:
            self.adp_last = prepared["record"]
        hidden = self.run_language_model(prepared)
        return BatchFeature(
            data={
                "backbone_features": hidden,
                "backbone_attention_mask": prepared["attention_mask"] == 1,
                "image_mask": prepared["image_mask"],
            }
        )  # [B, T2, hidden_size]

    def prepare_pruned_inputs(self, vl_input: BatchFeature, apply_drop: bool = True) -> dict:
        """Builds the language-model inputs; with ``apply_drop`` the pruned tokens are removed.

        ``keep_mask`` (B, L) over the full sequence is returned too (``None`` if nothing is
        dropped), so tests can rebuild the full-length reference.
        """
        input_ids = vl_input["input_ids"]
        attention_mask = vl_input["attention_mask"]
        pixel_values = vl_input["pixel_values"]
        image_grid_thw = vl_input["image_grid_thw"]
        core = self.model.model  # Qwen3VLModel

        inputs_embeds = core.get_input_embeddings()(input_ids)
        image_embeds, deepstack_embeds = core.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = core.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        image_mask = image_mask[..., 0]
        # MRoPE ids on the full sequence, so kept tokens retain their original (t, h, w) positions.
        position_ids, _ = core.get_rope_index(
            input_ids, image_grid_thw, None, attention_mask=attention_mask
        )

        keep_mask, record = self._select(
            inputs_embeds, input_ids, attention_mask, image_mask, image_grid_thw
        )
        prepared = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "image_mask": image_mask,
            "deepstack_embeds": deepstack_embeds,
            "keep_mask": keep_mask,
            "record": record,
        }
        if keep_mask is None or not apply_drop:
            return prepared

        n_keep = keep_mask.sum(dim=1)
        assert torch.all(n_keep == n_keep[0]), "every sample must keep the same token count"
        idx = keep_mask.nonzero()[:, 1].view(keep_mask.shape[0], -1)  # (B, L'), ascending
        prepared["inputs_embeds"] = torch.gather(
            inputs_embeds, 1, idx[..., None].expand(-1, -1, inputs_embeds.shape[-1])
        )
        prepared["attention_mask"] = torch.gather(attention_mask, 1, idx)
        prepared["position_ids"] = torch.gather(position_ids, 2, idx[None].expand(3, -1, -1))
        # DeepStack features are flattened in (batch, position) order of the image tokens.
        flat_keep = keep_mask[image_mask]
        prepared["deepstack_embeds"] = [d[flat_keep] for d in deepstack_embeds]
        prepared["image_mask"] = torch.gather(image_mask, 1, idx)
        return prepared

    def run_language_model(self, prepared: dict, attention_mask=None) -> torch.Tensor:
        """Output of the last decoder layer, before the final norm.

        This matches the original ``self.model(..., output_hidden_states=True).hidden_states[-1]``:
        through ``Qwen3VLForConditionalGeneration`` transformers does not tie the last entry to
        the normed ``last_hidden_state``, whereas calling the language model directly would.
        """
        language_model = self.model.model.language_model
        captured = {}
        handle = language_model.layers[-1].register_forward_hook(
            lambda _module, _args, output: captured.__setitem__("hidden", output)
        )
        try:
            language_model(
                input_ids=None,
                position_ids=prepared["position_ids"],
                attention_mask=prepared["attention_mask"]
                if attention_mask is None
                else attention_mask,
                inputs_embeds=prepared["inputs_embeds"],
                visual_pos_masks=prepared["image_mask"],
                deepstack_visual_embeds=prepared["deepstack_embeds"],
            )
        finally:
            handle.remove()
        return captured["hidden"]

    @torch.no_grad()
    def _select(self, inputs_embeds, input_ids, attention_mask, image_mask, image_grid_thw):
        """Returns (keep_mask over the full sequence or None, per-sample record)."""
        cfg = self.adp_config
        core = self.model.model
        batch_size = input_ids.shape[0]
        merge = core.visual.spatial_merge_size
        counts_all = (image_grid_thw.prod(-1) // merge**2).tolist()
        assert len(counts_all) % batch_size == 0, "samples must have the same number of images"
        n_images = len(counts_all) // batch_size
        counts = counts_all[:n_images]
        assert counts_all == counts * batch_size, "samples must have identical image layouts"

        views = self._image_view_names(n_images)
        weights = [cfg.view_weights.get(v, 1.0) for v in views] if views else None
        layer0 = core.language_model.layers[0]

        keep_mask = (
            None
            if cfg.keep_ratio >= 1.0
            else attention_mask.new_ones(attention_mask.shape, dtype=torch.bool)
        )
        record = {
            "counts": counts,
            "views": views,
            "grid_thw": image_grid_thw[:n_images].tolist(),
            "scores": [],
            "kept": [],
        }
        if keep_mask is None and not cfg.record:
            return None, record
        for b in range(batch_size):
            image_pos = image_mask[b].nonzero().squeeze(-1)
            text_pos = self._text_positions(input_ids[b], attention_mask[b], image_mask[b])
            scores = text_to_image_scores(layer0, inputs_embeds[b], text_pos, image_pos)
            per_image = list(torch.split(scores, counts))
            if keep_mask is None:
                kept = [torch.arange(n, device=scores.device) for n in counts]
            else:
                kept = select_tokens(
                    per_image, cfg.keep_ratio, cfg.budget_policy, weights, cfg.min_per_view
                )
                offsets = torch.tensor([0] + counts[:-1], device=scores.device).cumsum(0)
                kept_flat = torch.cat([k + o for k, o in zip(kept, offsets)])
                drop = torch.ones(image_pos.numel(), dtype=torch.bool, device=scores.device)
                drop[kept_flat] = False
                keep_mask[b, image_pos[drop]] = False
            if cfg.record:
                record["scores"].append([s.cpu() for s in per_image])
                record["kept"].append([k.cpu() for k in kept])
        return keep_mask, record

    def _text_positions(self, ids, mask, image_mask):
        valid = (mask == 1) & ~image_mask
        if self.adp_config.text_source == "all_non_image":
            return valid.nonzero().squeeze(-1)
        tok = self._special_ids()
        pos = torch.arange(ids.numel(), device=ids.device)
        last_vision_end = pos[ids == tok[VISION_END_TOKEN]].max()
        after = pos > last_vision_end
        im_end_after = pos[after & (ids == tok[IM_END_TOKEN])]
        end = im_end_after.min() if im_end_after.numel() else ids.numel()
        text = valid & after & (pos < end)
        assert text.any(), "no task text tokens found after the last <|vision_end|>"
        return text.nonzero().squeeze(-1)

    def _special_ids(self) -> dict[str, int]:
        if getattr(self, "_adp_special_ids", None) is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path)
            self._adp_special_ids = {
                t: tokenizer.convert_tokens_to_ids(t) for t in (VISION_END_TOKEN, IM_END_TOKEN)
            }
        return self._adp_special_ids

    def _image_view_names(self, n_images: int) -> list[str]:
        # Processor order is frame-major: (t0, v0), (t0, v1), ..., (t1, v0), ...
        keys = self.adp_config.view_keys
        if not keys or n_images % len(keys) != 0:
            return []
        return [keys[i % len(keys)] for i in range(n_images)]


def enable_adp(backbone: Qwen3Backbone, config: ADPConfig) -> Qwen3BackboneADP:
    """Switch a loaded ``Qwen3Backbone`` to the ADP forward in place, sharing its weights."""
    if not isinstance(backbone, Qwen3Backbone):
        raise TypeError(f"expected Qwen3Backbone, got {type(backbone).__name__}")
    backbone.__class__ = Qwen3BackboneADP
    backbone.adp_config = config
    backbone.adp_last = None
    return backbone


def disable_adp(backbone: Qwen3BackboneADP) -> Qwen3Backbone:
    """Restore the original ``Qwen3Backbone.forward``."""
    backbone.__class__ = Qwen3Backbone
    return backbone
