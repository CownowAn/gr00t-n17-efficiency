# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical checks of VLA-Cache for GR00T N1.7 on real trajectory frames (batch size 1).

1. Disabled, and the first inference of an episode (empty cache), equal the original model.
2. The attention statistics kept for the next inference (per-layer entropy, text -> image
   relevance) equal those of the real attention modules (eager, real MRoPE and mask).
3. Same frame twice: whatever is reused, the features equal the full computation (checks the
   partial-query attention, RoPE positions, masks, KV / hidden merge and DeepStack gather).
4. Frame A then frame B: the features equal an independent reference that runs every layer on
   all tokens through the original decoder layers and only swaps in the cached keys / values
   (and last-layer states) of the reused positions.
5. A changed prompt invalidates the cache; ``reset_cache`` empties it.

Run (CPU, fp32):
    CUDA_VISIBLE_DEVICES="" python -m gr00t_vlacache.tests.check_equivalence \
        --model-path <ckpt> --dataset-path <lerobot dataset>
"""

from dataclasses import dataclass

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t_adp.adp_policy import no_flash_attention
from gr00t_adp.data_utils import build_model_inputs, load_observations
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_vl import modeling_qwen3_vl as mq
import tyro

from gr00t_vlacache import VLACacheConfig
from gr00t_vlacache.backbone import disable_vlacache_backbone, enable_vlacache_backbone
from gr00t_vlacache.static_tokens import attention_entropy


@dataclass
class Args:
    model_path: str
    dataset_path: str
    embodiment_tag: str = "new_embodiment"
    traj_id: int = 0
    steps: tuple[int, ...] = (40, 56)
    device: str = "cpu"


TOL = 1e-4


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def fresh(backbone_output):
    """Shallow copy: the action head overwrites ``backbone_features`` in place."""
    return BatchFeature(data=dict(backbone_output))


class KVOverride:
    """Stand-in cache for the reference: swaps in cached K / V at the reused positions."""

    def __init__(self, cached_kv, positions):
        self.cached_kv, self.positions = cached_kv, positions

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.positions is not None:
            key_states, value_states = key_states.clone(), value_states.clone()
            k_old, v_old = self.cached_kv[layer_idx]
            key_states[:, :, self.positions] = k_old[:, :, self.positions]
            value_states[:, :, self.positions] = v_old[:, :, self.positions]
        self.last = (key_states, value_states)
        return key_states, value_states


class Checker:
    def __init__(self, policy: Gr00tPolicy):
        self.policy = policy
        self.model = policy.model
        image_processor = policy.processor.processor.image_processor
        self.base = dict(
            view_keys=list(policy.modality_configs["video"].modality_keys),
            image_mean=tuple(image_processor.image_mean),
            image_std=tuple(image_processor.image_std),
        )

    def enable(self, **kwargs):
        return enable_vlacache_backbone(self.model.backbone, VLACacheConfig(**self.base, **kwargs))

    def original(self, inputs):
        disable_vlacache_backbone(self.model.backbone)
        return Qwen3Backbone.forward(self.model.backbone, inputs)

    def action(self, out, action_inputs):
        torch.manual_seed(0)
        return self.model.action_head.get_action(fresh(out), action_inputs)["action_pred"]

    def compare(self, name, out, ref, action_inputs=None, extra=""):
        valid = ref["backbone_attention_mask"]
        ok = torch.equal(valid, out["backbone_attention_mask"])
        ok &= torch.equal(ref["image_mask"], out["image_mask"])
        err = max_rel(out["backbone_features"][valid], ref["backbone_features"][valid])
        detail = f"features rel err={err:.3e}"
        ok &= err < TOL
        if action_inputs is not None:
            err_a = max_rel(self.action(out, action_inputs), self.action(ref, action_inputs))
            detail += f", action_pred rel err={err_a:.3e}"
            ok &= err_a < TOL
        return report(name, ok, detail + extra)

    # 1 -----------------------------------------------------------------------------------
    def check_off_and_first(self, inputs, action_inputs) -> bool:
        ref = self.original(inputs)
        ok = self.compare(
            "disabled == original", self.enable(enabled=False)(inputs), ref, action_inputs
        )
        backbone = self.enable()
        out = backbone(inputs)
        ok &= self.compare("first inference (empty cache) == original", out, ref, action_inputs)
        return ok

    # 2 -----------------------------------------------------------------------------------
    def check_statistics(self, inputs) -> bool:
        backbone = self.enable()
        layers = backbone.model.model.language_model.layers
        captured = {}

        def make_hook(i):
            def hook(_m, args, kwargs):
                captured[i] = (args[0].clone(), kwargs["position_embeddings"])

            return hook

        handles = [
            layer.register_forward_pre_hook(make_hook(i), with_kwargs=True)
            for i, layer in enumerate(layers)
        ]
        try:
            backbone(inputs)
        finally:
            for h in handles:
                h.remove()
        cache = backbone.vlac_cache

        mask = inputs["attention_mask"]
        length = mask.shape[1]
        allowed = torch.ones(length, length, dtype=torch.bool).tril() & (mask[0][None, :] == 1)
        prepared = backbone.prepare_pruned_inputs(inputs)
        image_mask = prepared["image_mask"][0]
        text_pos = backbone._text_positions(inputs["input_ids"][0], mask[0], image_mask)
        image_pos = image_mask.nonzero().squeeze(-1)

        worst_ent, ref_rel = 0.0, None
        for i, layer in enumerate(layers):
            hidden, (cos, sin) = captured[i]
            additive = torch.zeros(1, 1, length, length, dtype=hidden.dtype)
            additive = additive.masked_fill(~allowed, torch.finfo(hidden.dtype).min)
            attn = layer.self_attn
            previous = attn.config._attn_implementation
            attn.config._attn_implementation = "eager"
            try:
                _, weights = mq.Qwen3VLTextAttention.forward(
                    attn, layer.input_layernorm(hidden), (cos, sin), attention_mask=additive
                )
            finally:
                attn.config._attn_implementation = previous
            worst_ent = max(
                worst_ent, max_rel(cache["entropy"][i : i + 1], attention_entropy(weights[0])[None])
            )
            if i == backbone.vlac_config.relevance_layer:
                ref_rel = weights[0][:, text_pos][:, :, image_pos].float().mean(0).mean(0)
        err_rel = max_rel(cache["relevance"], ref_rel)
        rank_ok = torch.equal(cache["relevance"].argsort(), ref_rel.argsort())
        return report(
            "attention statistics == eager attention modules",
            worst_ent < TOL and err_rel < TOL and rank_ok,
            f"entropy rel err={worst_ent:.3e}, relevance rel err={err_rel:.3e}, same ranking={rank_ok}",
        )

    def reuse_detail(self, backbone) -> str:
        rec = backbone.vlac_last
        plan = rec["plan"]
        if plan["reusable"] is None:
            return ", nothing reused"
        static = [len(v["static"]) for v in plan["per_view"]]
        per_layer = {}
        for i, r in enumerate(rec["reused_per_layer"]):
            n = 0 if r is None else r.numel()
            if i in backbone.vlac_config.reuse_layers:
                per_layer[i] = n
        return (
            f", static per view {static} of {plan['counts']}, reusable {plan['reusable'].numel()}, "
            f"reused from layer {per_layer}"
        )

    # 3 -----------------------------------------------------------------------------------
    def check_same_frame(self, inputs, action_inputs) -> bool:
        ref = self.original(inputs)
        backbone = self.enable(record=True)
        backbone(inputs)
        out = backbone(inputs)
        return self.compare(
            "same frame twice == full computation",
            out,
            ref,
            action_inputs,
            self.reuse_detail(backbone),
        )

    # 4 -----------------------------------------------------------------------------------
    def check_two_frames(self, inputs_a, inputs_b) -> bool:
        backbone = self.enable(record=True)
        backbone(inputs_a)
        cache_a = backbone.vlac_cache
        out = backbone(inputs_b)
        detail = self.reuse_detail(backbone)

        backbone.vlac_cache = cache_a
        prepared = backbone.prepare_pruned_inputs(inputs_b)
        plan = backbone.plan_reuse(prepared, inputs_b)
        hidden, _ = backbone.run_layers(
            prepared, inputs_b, plan, kv_override=lambda idx, pos: KVOverride(cache_a["kv"], pos)
        )
        ref = {
            "backbone_features": hidden,
            "backbone_attention_mask": prepared["attention_mask"] == 1,
            "image_mask": prepared["image_mask"],
        }
        full = self.original(inputs_b)
        drift = max_rel(out["backbone_features"], full["backbone_features"])
        return self.compare(
            "frame A -> B == KV-swap reference",
            out,
            ref,
            None,
            detail + f", vs full recompute rel diff={drift:.3e}",
        )

    # 5 -----------------------------------------------------------------------------------
    def check_invalidation(self, inputs, inputs_other_text) -> bool:
        backbone = self.enable(record=True)
        backbone(inputs)
        out = backbone(inputs_other_text)
        ok = backbone.vlac_last["plan"]["reusable"] is None
        ok &= self.compare("prompt change -> full recompute", out, self.original(inputs_other_text))
        backbone = self.enable(record=True)
        backbone(inputs)
        backbone.reset_cache()
        backbone(inputs)
        ok &= backbone.vlac_last["plan"]["reusable"] is None
        return report("invalidation and reset", ok, "no reuse after prompt change or reset")


def main(args: Args):
    with no_flash_attention():
        policy = Gr00tPolicy(args.embodiment_tag, args.model_path, device=args.device)
    policy.model.to(torch.float32)
    checker = Checker(policy)
    print(f"views={checker.base['view_keys']}")

    obs_a, obs_b = load_observations(policy, args.dataset_path, args.traj_id, list(args.steps))
    other = load_observations(policy, args.dataset_path, args.traj_id, [args.steps[0]])[0]
    lang_key = next(iter(other["language"]))
    other["language"][lang_key] = [[other["language"][lang_key][0][0] + " carefully"]]

    all_ok = True
    with torch.inference_mode():
        inputs_a, action_a = build_model_inputs(policy, obs_a, dtype=torch.float32)
        inputs_b, _ = build_model_inputs(policy, obs_b, dtype=torch.float32)
        inputs_o, _ = build_model_inputs(policy, other, dtype=torch.float32)
        print(f"sequence {tuple(inputs_a['input_ids'].shape)}, steps {args.steps}")
        all_ok &= checker.check_off_and_first(inputs_a, action_a)
        all_ok &= checker.check_statistics(inputs_a)
        all_ok &= checker.check_same_frame(inputs_a, action_a)
        all_ok &= checker.check_two_frames(inputs_a, inputs_b)
        all_ok &= checker.check_invalidation(inputs_a, inputs_o)
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main(tyro.cli(Args))
