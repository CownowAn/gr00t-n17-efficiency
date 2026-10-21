# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical checks of EfficientVLA for GR00T N1.7 against independent references on real data.

1. All components off: backbone features and action_pred equal the original model.
2. Relevance scores equal the attention weights of the real layer (eager, real MRoPE + mask).
3. Layer pruning equals the original backbone with those decoder layers made identity.
4. Token pruning (gather) equals keeping the full sequence and masking the dropped tokens as
   keys from layer ``prune_after`` on, for several prune points / layer drops.
5. DiT caching equals the original action head with attn / MLP outputs replaced by forward
   hooks on reuse steps; attention is called only on compute steps.
6. Everything on together runs and keeps the per-view budgets.

Run (CPU, fp32):
    CUDA_VISIBLE_DEVICES="" python -m gr00t_efficientvla.tests.check_equivalence \
        --model-path <ckpt> --dataset-path <lerobot dataset>
"""

from dataclasses import dataclass
import math

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t_adp.adp_policy import no_flash_attention
from gr00t_adp.data_utils import build_model_inputs, load_observations, stack_observations
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_vl import modeling_qwen3_vl as mq
import tyro

from gr00t_efficientvla import EfficientVLAConfig, enable_efficientvla
from gr00t_efficientvla.backbone import disable_efficientvla_backbone
from gr00t_efficientvla.dit_cache import denoising_step, disable_dit_cache


@dataclass
class Args:
    model_path: str
    dataset_path: str
    embodiment_tag: str = "new_embodiment"
    traj_id: int = 0
    steps: tuple[int, ...] = (0, 40)
    keep_ratio: float = 0.5
    device: str = "cpu"


TOL = 1e-4


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def fresh(backbone_output):
    """Shallow copy: the action head overwrites ``backbone_features`` in place."""
    return BatchFeature(data=dict(backbone_output))


def max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


class Checker:
    def __init__(self, policy: Gr00tPolicy):
        self.policy = policy
        self.model = policy.model
        self.view_keys = list(policy.modality_configs["video"].modality_keys)

    def config(self, **kwargs) -> EfficientVLAConfig:
        return EfficientVLAConfig.disabled(view_keys=self.view_keys, **kwargs)

    def enable(self, config):
        enable_efficientvla(self.model, config)
        return self.model.backbone

    def disable(self):
        disable_efficientvla_backbone(self.model.backbone)
        disable_dit_cache(self.model.action_head)

    def original(self, inputs, action_inputs):
        self.disable()
        out = Qwen3Backbone.forward(self.model.backbone, inputs)
        torch.manual_seed(0)
        return out, self.model.action_head.get_action(fresh(out), action_inputs)["action_pred"]

    def evla(self, config, inputs, action_inputs):
        backbone = self.enable(config)
        out = backbone(inputs)
        torch.manual_seed(0)
        return out, self.model.action_head.get_action(fresh(out), action_inputs)["action_pred"]

    # 1 -----------------------------------------------------------------------------------
    def check_disabled(self, inputs, action_inputs) -> bool:
        ref, ref_act = self.original(inputs, action_inputs)
        out, act = self.evla(self.config(), inputs, action_inputs)
        valid = ref["backbone_attention_mask"]
        ok = torch.equal(valid, out["backbone_attention_mask"])
        ok &= torch.equal(ref["image_mask"], out["image_mask"])
        err_f = max_rel(out["backbone_features"][valid], ref["backbone_features"][valid])
        err_a = max_rel(act, ref_act)
        return report(
            "all off == original",
            ok and err_f < TOL and err_a < TOL,
            f"features rel err={err_f:.3e}, action_pred rel err={err_a:.3e}",
        )

    # 2 -----------------------------------------------------------------------------------
    def check_scores(self, inputs, prune_after=2) -> bool:
        backbone = self.enable(self.config(keep_ratio=0.5, prune_after=prune_after))
        layer = backbone.model.model.language_model.layers[prune_after - 1]
        captured = {}

        def pre_hook(_module, args, kwargs):
            captured["hidden"] = args[0].clone()
            captured["pos"] = kwargs["position_embeddings"]

        relevance = backbone._relevance
        backbone._relevance = lambda lyr, st: captured.setdefault("ours", relevance(lyr, st))
        handle = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            backbone(inputs)
        finally:
            handle.remove()
            del backbone._relevance

        hidden, (cos, sin) = captured["hidden"], captured["pos"]
        mask = inputs["attention_mask"]
        length = mask.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=mask.device).tril()
        allowed = causal[None] & (mask[:, None, :] == 1)
        additive = torch.zeros(allowed.shape, dtype=hidden.dtype, device=hidden.device)
        additive = additive.masked_fill(~allowed, torch.finfo(hidden.dtype).min)[:, None]
        attn = layer.self_attn
        previous = attn.config._attn_implementation
        attn.config._attn_implementation = "eager"
        try:
            _, weights = mq.Qwen3VLTextAttention.forward(
                attn, layer.input_layernorm(hidden), (cos, sin), attention_mask=additive
            )
        finally:
            attn.config._attn_implementation = previous

        worst, rank_ok = 0.0, True
        image_mask = backbone.prepare_pruned_inputs(inputs)["image_mask"]
        for b, ours in enumerate(captured["ours"]):
            text_pos = backbone._text_positions(inputs["input_ids"][b], mask[b], image_mask[b])
            image_pos = image_mask[b].nonzero().squeeze(-1)
            ref = weights[b][:, text_pos][:, :, image_pos].float().mean(0).sum(0)
            worst = max(worst, max_rel(ours, ref))
            rank_ok &= torch.equal(ours.argsort(), ref.argsort())
        return report(
            f"relevance == layer-{prune_after - 1} attention weights",
            worst < TOL and rank_ok,
            f"rel err={worst:.3e}, same ranking={rank_ok}",
        )

    # 3 -----------------------------------------------------------------------------------
    def check_layer_drop(self, inputs, drop_layers) -> bool:
        layers = self.model.backbone.model.model.language_model.layers
        self.disable()
        for i in drop_layers:
            layers[i].forward = lambda hidden_states, *a, **k: hidden_states
        try:
            ref = Qwen3Backbone.forward(self.model.backbone, inputs)
        finally:
            for i in drop_layers:
                del layers[i].forward
        out = self.enable(self.config(drop_layers=drop_layers))(inputs)
        valid = ref["backbone_attention_mask"]
        err = max_rel(out["backbone_features"][valid], ref["backbone_features"][valid])
        return report(
            f"drop layers {tuple(drop_layers)} == identity layers",
            err < TOL,
            f"rel err={err:.3e}",
        )

    # 4 -----------------------------------------------------------------------------------
    def check_token_drop(self, inputs, keep_ratio, prune_after, drop_layers=()) -> bool:
        cfg = self.config(keep_ratio=keep_ratio, prune_after=prune_after, drop_layers=drop_layers)
        backbone = self.enable(cfg)
        out = backbone(inputs)
        prepared = backbone.prepare_pruned_inputs(inputs)
        ref_hidden, ref_state = backbone.run_layers(prepared, emulate_drop=True)
        keep = ref_state["keep"]
        feats = out["backbone_features"]
        ok, worst = True, 0.0
        for b in range(feats.shape[0]):
            ref_b = ref_hidden[b][keep[b]]
            valid = out["backbone_attention_mask"][b]
            ok &= ref_b.shape == feats[b].shape
            worst = max(worst, max_rel(feats[b][valid], ref_b[valid]))
        n_before = int(prepared["image_mask"][0].sum())
        n_after = int(out["image_mask"][0].sum())
        return report(
            f"token drop == masked (keep={keep_ratio}, prune_after={prune_after}, "
            f"drop={tuple(drop_layers)})",
            ok and worst < TOL,
            f"image tokens {n_before}->{n_after}, seq {keep.shape[1]}->{feats.shape[1]}, "
            f"rel err={worst:.3e}",
        )

    # 5 -----------------------------------------------------------------------------------
    def check_dit_cache(self, inputs, action_inputs, interval) -> bool:
        head = self.model.action_head
        dit = head.model
        n_steps, n_buckets = head.num_inference_timesteps, head.num_timestep_buckets
        out, _ = self.original(inputs, action_inputs)

        # Reference: original blocks; on reuse steps forward hooks return the cached outputs.
        state, cache, handles = {}, {}, []

        def pre_hook(_m, _a, kwargs):
            step = denoising_step(kwargs["timestep"], n_steps, n_buckets)
            state["compute"] = step % interval == 0

        handles.append(dit.register_forward_pre_hook(pre_hook, with_kwargs=True))
        for i, block in enumerate(dit.transformer_blocks):
            for name in ("attn1", "ff"):

                def hook(_m, _a, output, key=(i, name)):
                    if state["compute"]:
                        cache[key] = output
                        return None
                    return cache[key]

                handles.append(getattr(block, name).register_forward_hook(hook))
        try:
            torch.manual_seed(0)
            ref = head.get_action(fresh(out), action_inputs)["action_pred"]
        finally:
            for h in handles:
                h.remove()

        self.enable(self.config(cache_interval=interval))
        calls = []
        handles = [
            b.attn1.register_forward_hook(lambda *_: calls.append(1))
            for b in dit.transformer_blocks
        ]
        try:
            torch.manual_seed(0)
            ours = head.get_action(fresh(out), action_inputs)["action_pred"]
        finally:
            for h in handles:
                h.remove()
        expected_calls = len(dit.transformer_blocks) * math.ceil(n_steps / interval)
        err = max_rel(ours, ref)
        return report(
            f"DiT cache interval={interval} == hook reference",
            err < TOL and len(calls) == expected_calls,
            f"rel err={err:.3e}, attn calls {len(calls)} (expected {expected_calls}, "
            f"uncached {len(dit.transformer_blocks) * n_steps})",
        )

    # 6 -----------------------------------------------------------------------------------
    def check_all_on(self, inputs, action_inputs, keep_ratio) -> bool:
        n_layers = len(self.model.backbone.model.model.language_model.layers)
        drop = tuple(range(n_layers - 4, n_layers))  # placeholder ranking; calibration picks real
        cfg = self.config(drop_layers=drop, keep_ratio=keep_ratio, cache_interval=2, record=True)
        out, act = self.evla(cfg, inputs, action_inputs)
        rec = self.model.backbone.evla_last
        kept = [[len(s["kept"]) for s in sample] for sample in rec["selections"]]
        expected = [max(1, round(n * keep_ratio)) for n in rec["counts"]]
        ok = all(k == expected for k in kept) and bool(torch.isfinite(act).all())
        ok &= int(out["image_mask"][0].sum()) == sum(expected)
        return report(
            "all on (drop last 4 layers, tokens, cache=2)",
            ok,
            f"kept per view {kept[0]} of {rec['counts']}, action finite={bool(torch.isfinite(act).all())}",
        )


def main(args: Args):
    with no_flash_attention():
        policy = Gr00tPolicy(args.embodiment_tag, args.model_path, device=args.device)
    policy.model.to(torch.float32)
    checker = Checker(policy)
    n_layers = len(policy.model.backbone.model.model.language_model.layers)
    print(f"views={checker.view_keys}, LLM layers={n_layers}")

    observations = load_observations(policy, args.dataset_path, args.traj_id, list(args.steps))
    second = observations[1]
    lang_key = next(iter(second["language"]))
    second["language"][lang_key] = [[second["language"][lang_key][0][0] + " carefully and quickly"]]
    batches = {"B=1": observations[0], "B=2 (padded)": stack_observations(observations[:2])}

    all_ok = True
    with torch.inference_mode():
        for name, obs in batches.items():
            print(f"--- {name}")
            inputs, action_inputs = build_model_inputs(policy, obs, dtype=torch.float32)
            print(f"sequence {tuple(inputs['input_ids'].shape)}")
            all_ok &= checker.check_disabled(inputs, action_inputs)
            all_ok &= checker.check_scores(inputs, prune_after=2)
            all_ok &= checker.check_layer_drop(inputs, (1, n_layers // 2, n_layers - 1))
            for prune_after, drop in [(2, ()), (1, (2, n_layers // 2)), (4, (1,))]:
                all_ok &= checker.check_token_drop(inputs, args.keep_ratio, prune_after, drop)
            for interval in (1, 2, 3):
                all_ok &= checker.check_dit_cache(inputs, action_inputs, interval)
            all_ok &= checker.check_all_on(inputs, action_inputs, args.keep_ratio)
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main(tyro.cli(Args))
