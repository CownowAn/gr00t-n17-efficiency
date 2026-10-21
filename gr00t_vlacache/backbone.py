# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3 backbone with VLA-Cache: reuse the KV of static, task-irrelevant image tokens across
inferences (arXiv 2502.02175; reference: siyuhsu/vla-cache, OpenVLA-OFT branch).

Per inference (batch size 1), with the cache built by the previous inference of the episode:

1. static tokens: per view, merged tokens whose raw pixels have cosine similarity >= threshold
   to the frame the cache was built from (the reference OFT code compares the current frame with
   itself; here the previous inference's frame is used, as intended);
2. minus the most attended tokens of the previous inference (text -> image, ``relevance_layer``);
3. at each layer in ``reuse_layers`` a prefix of these tokens, sized by the per-layer reuse
   proportion (1 - normalized attention entropy of the previous inference), stops being
   computed: its queries are dropped and its keys / values come from the cache;
4. GR00T's action head reads every token's last-layer state, so reused tokens also take their
   last-layer hidden state from the cache.

Layers that still see every token run through the original decoder layer (FlashAttention on
GPU); layers with reused tokens use SDPA with the computed queries against the full key set.
The attention statistics for the next inference are recomputed exactly per layer, since
FlashAttention returns no weights.
"""

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
from gr00t_adp.config import ADPConfig
from gr00t_adp.qwen3_backbone_adp import Qwen3BackboneADP
import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.feature_extraction_utils import BatchFeature
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

from .config import VLACacheConfig
from .static_tokens import (
    attention_entropy,
    layer_reuse_schedule,
    relevant_tokens,
    reusable_tokens,
    static_tokens,
    token_pixels,
    tokens_to_reuse,
)


class Qwen3BackboneVLACache(Qwen3BackboneADP):
    vlac_config: VLACacheConfig
    vlac_cache: dict | None
    vlac_last: dict | None

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        if vl_input["input_ids"].shape[0] != 1:
            raise ValueError("VLA-Cache supports batch size 1 only")
        prepared = self.prepare_pruned_inputs(vl_input)
        if not self.vlac_config.enabled:
            self.vlac_cache = None
            hidden = self.run_language_model(prepared)
        else:
            plan = self.plan_reuse(prepared, vl_input)
            hidden, self.vlac_cache = self.run_layers(prepared, vl_input, plan)
        return BatchFeature(
            data={
                "backbone_features": hidden,
                "backbone_attention_mask": prepared["attention_mask"] == 1,
                "image_mask": prepared["image_mask"],
            }
        )  # [B, T2, hidden_size]

    def reset_cache(self) -> None:
        self.vlac_cache = None
        self.vlac_last = None

    # ------------------------------------------------------------------ selection
    def _counts(self, image_grid_thw) -> list[int]:
        merge = self.model.model.visual.spatial_merge_size
        return (image_grid_thw.prod(-1) // merge**2).tolist()

    def _cache_matches(self, vl_input) -> bool:
        cache = self.vlac_cache
        return (
            cache is not None
            and torch.equal(cache["input_ids"], vl_input["input_ids"])
            and torch.equal(cache["attention_mask"], vl_input["attention_mask"])
            and torch.equal(cache["image_grid_thw"], vl_input["image_grid_thw"])
        )

    @torch.no_grad()
    def plan_reuse(self, prepared, vl_input) -> dict:
        """Reusable sequence positions (ascending) and the per-layer reuse proportions."""
        cfg = self.vlac_config
        counts = self._counts(vl_input["image_grid_thw"])
        merge = self.model.model.visual.spatial_merge_size
        pixels = token_pixels(
            vl_input["pixel_values"], counts, merge, cfg.image_mean, cfg.image_std
        )
        plan = {"pixels": pixels, "counts": counts, "reusable": None, "schedule": None}
        if not self._cache_matches(vl_input):
            return plan

        cache = self.vlac_cache
        image_pos = prepared["image_mask"][0].nonzero().squeeze(-1)
        offset, reusable, per_view = 0, [], []
        for i, n in enumerate(counts):
            static = static_tokens(
                pixels[i], cache["pixels"][i], cfg.sim_threshold, cfg.static_ratio
            )
            relevant = relevant_tokens(cache["relevance"][offset : offset + n], cfg.relevant_ratio)
            local = reusable_tokens(static, relevant)
            reusable.append(image_pos[offset + local])
            per_view.append(
                {"static": static.cpu(), "relevant": relevant.cpu(), "reusable": local.cpu()}
            )
            offset += n
        plan["reusable"] = torch.cat(reusable)
        plan["schedule"] = layer_reuse_schedule(cache["entropy"], cfg.growth_factor)
        plan["per_view"] = per_view
        return plan

    # ------------------------------------------------------------------ decoder loop
    def run_layers(self, prepared, vl_input, plan, kv_override=None):
        """Decoder loop with token reuse. Returns (last-layer hidden of every token, new cache).

        ``kv_override`` is only for tests: if given, every layer runs on all tokens and the
        listed positions' keys / values are replaced by the cached ones (the reference).
        """
        cfg = self.vlac_config
        lm = self.model.model.language_model
        cache = self.vlac_cache
        hidden = prepared["inputs_embeds"]
        length = hidden.shape[1]
        device = hidden.device
        valid = prepared["attention_mask"][0] == 1
        position_ids = prepared["position_ids"]
        cache_position = torch.arange(length, device=device)
        cos, sin = lm.rotary_emb(hidden, position_ids)
        full_mask = create_causal_mask(
            config=lm.config,
            input_embeds=hidden,
            attention_mask=prepared["attention_mask"],
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids[0],
        )
        image_mask = prepared["image_mask"][0]
        image_index = torch.cumsum(image_mask.long(), 0) - 1  # position -> row of image features
        deepstack = prepared["deepstack_embeds"]
        text_pos = self._text_positions(
            vl_input["input_ids"][0], prepared["attention_mask"][0], image_mask
        )

        comp = cache_position  # positions still computed
        last = None
        kv_new, entropies, relevance = [], [], None
        record = {"reused_per_layer": [], "plan": plan}
        full_cache = DynamicCache()
        for idx, layer in enumerate(lm.layers):
            if plan["reusable"] is not None and idx in cfg.reuse_layers:
                selected = tokens_to_reuse(plan["reusable"], float(plan["schedule"][idx]))
                if selected.numel() and (last is None or last.numel() <= selected.numel()):
                    keep = ~torch.isin(comp, selected)
                    if kv_override is None:
                        hidden = hidden[:, keep]
                        comp = comp[keep]
                    last = selected
            record["reused_per_layer"].append(last.cpu() if last is not None else None)
            layer_input = hidden

            if kv_override is not None:
                override = kv_override(idx, last)
                hidden = layer(
                    hidden,
                    attention_mask=full_mask,
                    position_ids=position_ids[0],
                    past_key_values=override,
                    cache_position=cache_position,
                    position_embeddings=(cos, sin),
                )
                k_full, v_full = override.last
            elif comp.numel() == length:
                hidden = layer(
                    hidden,
                    attention_mask=full_mask,
                    position_ids=position_ids[0],
                    past_key_values=full_cache,
                    cache_position=cache_position,
                    position_embeddings=(cos, sin),
                )
                k_full, v_full = full_cache.layers[idx].keys, full_cache.layers[idx].values
            else:
                hidden, k_full, v_full = self._partial_layer(
                    layer, hidden, comp, cos, sin, valid, cache["kv"][idx]
                )
            kv_new.append((k_full, v_full))

            rows = cache_position if kv_override is not None else comp
            probs = self._attention_probs(layer, layer_input, rows, k_full, cos, sin, valid)
            entropies.append(attention_entropy(probs))
            if idx == cfg.relevance_layer:
                text_rows = torch.searchsorted(rows, text_pos)
                image_pos = image_mask.nonzero().squeeze(-1)
                relevance = probs[:, text_rows][:, :, image_pos].float().mean(dim=0).mean(dim=0)

            if idx < len(deepstack):
                is_image = image_mask[rows]
                hidden = hidden.clone()
                hidden[0, is_image] = hidden[0, is_image] + deepstack[idx][
                    image_index[rows[is_image]]
                ].to(hidden.dtype)

        if kv_override is not None or comp.numel() == length:
            final = hidden
        else:
            final = cache["hidden"].clone()
            final[:, comp] = hidden
        if kv_override is not None and last is not None:
            final = final.clone()
            final[:, last] = cache["hidden"][:, last]

        self.vlac_last = record if cfg.record else None
        new_cache = {
            "input_ids": vl_input["input_ids"],
            "attention_mask": vl_input["attention_mask"],
            "image_grid_thw": vl_input["image_grid_thw"],
            "pixels": plan["pixels"],
            "kv": kv_new,
            "hidden": final,
            "entropy": torch.stack(entropies).cpu(),
            "relevance": relevance,
        }
        return final, new_cache

    @staticmethod
    def _project_qk(layer, x, rows, cos, sin):
        attn = layer.self_attn
        shape = (1, rows.numel(), -1, attn.head_dim)
        h = layer.input_layernorm(x)
        q = attn.q_norm(attn.q_proj(h).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(h).view(shape)).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos[:, rows], sin[:, rows])
        return h, q, k

    def _partial_layer(self, layer, x, comp, cos, sin, valid, cached_kv):
        """One decoder layer on the computed tokens; reused positions keep the cached K / V."""
        attn = layer.self_attn
        h, q, k = self._project_qk(layer, x, comp, cos, sin)
        v = attn.v_proj(h).view(1, comp.numel(), -1, attn.head_dim).transpose(1, 2)
        k_full = cached_kv[0].index_copy(2, comp, k)
        v_full = cached_kv[1].index_copy(2, comp, v)
        positions = torch.arange(k_full.shape[2], device=x.device)
        allowed = (positions[None, :] <= comp[:, None]) & valid[None, :]
        out = F.scaled_dot_product_attention(
            q,
            repeat_kv(k_full, attn.num_key_value_groups),
            repeat_kv(v_full, attn.num_key_value_groups),
            attn_mask=allowed[None, None],
            scale=attn.scaling,
        )
        out = attn.o_proj(out.transpose(1, 2).reshape(1, comp.numel(), -1))
        x = x + out
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return x, k_full, v_full

    def _attention_probs(self, layer, x, rows, k_full, cos, sin, valid):
        """(heads, rows, keys) float32 attention probabilities of ``layer`` for ``rows``."""
        attn = layer.self_attn
        _, q, _ = self._project_qk(layer, x, rows, cos, sin)
        k = repeat_kv(k_full, attn.num_key_value_groups)
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2))[0] * attn.scaling
        positions = torch.arange(k.shape[2], device=x.device)
        allowed = (positions[None, :] <= rows[:, None]) & valid[None, :]
        return logits.masked_fill(~allowed, float("-inf")).softmax(dim=-1)


def enable_vlacache_backbone(
    backbone: Qwen3Backbone, config: VLACacheConfig
) -> Qwen3BackboneVLACache:
    """Switch a loaded ``Qwen3Backbone`` to the VLA-Cache forward in place (cache emptied)."""
    if not isinstance(backbone, Qwen3Backbone):
        raise TypeError(f"expected Qwen3Backbone, got {type(backbone).__name__}")
    n_layers = len(backbone.model.model.language_model.layers)
    bad = [i for i in config.reuse_layers if i >= n_layers]
    if bad or not 0 <= config.relevance_layer < n_layers:
        raise ValueError(f"reuse_layers / relevance_layer out of range for {n_layers} layers")
    backbone.__class__ = Qwen3BackboneVLACache
    backbone.vlac_config = config
    # Bookkeeping helpers inherited from the ADP backbone read this; its pruning stays off.
    backbone.adp_config = ADPConfig(keep_ratio=1.0, view_keys=config.view_keys)
    backbone.adp_last = None
    backbone.reset_cache()
    return backbone


def disable_vlacache_backbone(backbone: Qwen3BackboneVLACache) -> Qwen3Backbone:
    backbone.__class__ = Qwen3Backbone
    return backbone
