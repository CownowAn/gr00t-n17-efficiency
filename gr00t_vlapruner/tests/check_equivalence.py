# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks of VLA-Pruner for GR00T N1.7 on consecutive trajectory frames (batch 1, CPU fp32).

1. Disabled, and the first ``history_window`` inferences of an episode (unpruned), equal the
   original model (features and action_pred).
2. Semantic relevance equals the real layer-K attention weights (eager, real MRoPE and mask),
   averaged over heads and all valid queries.
3. The recomputed DiT action -> key attention reproduces each hooked block's attention output
   (probabilities @ values -> to_out), and the recorded score is their mean.
4. Pruned inference: dropping equals keeping every token and masking the dropped ones as keys
   from layer K+1 on; budgets per view; kept tokens come from the semantic / action candidates.
5. History entries are full length with zeros at pruned tokens; ``reset_history`` empties it.

    CUDA_VISIBLE_DEVICES="" python -m gr00t_vlapruner.tests.check_equivalence \
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

from gr00t_vlapruner import VLAPrunerConfig
from gr00t_vlapruner.action_attention import cross_attention_probs
from gr00t_vlapruner.backbone import disable_vlapruner, enable_vlapruner


@dataclass
class Args:
    model_path: str
    dataset_path: str
    embodiment_tag: str = "new_embodiment"
    traj_id: int = 0
    steps: tuple[int, ...] = (0, 8, 16, 24, 32)
    keep_ratio: float = 0.25


TOL = 1e-4


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def fresh(backbone_output):
    """Shallow copy: the action head overwrites ``backbone_features`` in place."""
    return BatchFeature(data=dict(backbone_output))


class Checker:
    def __init__(self, policy: Gr00tPolicy, keep_ratio: float):
        self.policy = policy
        self.model = policy.model
        self.view_keys = list(policy.modality_configs["video"].modality_keys)
        self.keep_ratio = keep_ratio

    def config(self, **kwargs):
        return VLAPrunerConfig(keep_ratio=self.keep_ratio, view_keys=self.view_keys, **kwargs)

    def original(self, inputs, action_inputs):
        disable_vlapruner(self.model)
        out = Qwen3Backbone.forward(self.model.backbone, inputs)
        torch.manual_seed(0)
        return out, self.model.action_head.get_action(fresh(out), action_inputs)["action_pred"]

    def infer(self, inputs, action_inputs):
        out = self.model.backbone(inputs)
        torch.manual_seed(0)
        return out, self.model.action_head.get_action(fresh(out), action_inputs)["action_pred"]

    # 1 -----------------------------------------------------------------------------------
    def check_unpruned(self, frames) -> bool:
        refs = [self.original(*f) for f in frames]
        ok = True
        enable_vlapruner(self.model, self.config(enabled=False))
        out, act = self.infer(*frames[0])
        err = max(
            max_rel(out["backbone_features"], refs[0][0]["backbone_features"]),
            max_rel(act, refs[0][1]),
        )
        ok &= report("disabled == original", err < TOL, f"max rel err={err:.3e}")

        cfg = self.config()
        enable_vlapruner(self.model, cfg)
        worst = 0.0
        for f, (ref_out, ref_act) in zip(frames[: cfg.history_window], refs):
            out, act = self.infer(*f)
            same_len = out["backbone_features"].shape == ref_out["backbone_features"].shape
            worst = max(
                worst,
                max_rel(out["backbone_features"], ref_out["backbone_features"]),
                max_rel(act, ref_act),
            )
            ok &= same_len
        ok &= report(
            f"first {cfg.history_window} inferences (history filling) == original",
            ok and worst < TOL,
            f"max rel err={worst:.3e}, history length {len(self.model.backbone.vlap_history)}",
        )
        return ok

    # 2 -----------------------------------------------------------------------------------
    def check_semantic(self, inputs) -> bool:
        cfg = self.config()
        backbone = enable_vlapruner(self.model, cfg)
        layer = backbone.model.model.language_model.layers[cfg.prune_layer]
        captured = {}

        def pre_hook(_m, args, kwargs):
            captured["hidden"] = args[0].clone()
            captured["pos"] = kwargs["position_embeddings"]

        relevance = backbone._relevance
        handle = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            backbone(inputs)  # unpruned pass, just to reach layer K with the real inputs
            prepared = backbone.prepare_pruned_inputs(inputs)
            state = {
                "hidden": captured["hidden"],
                "attention_mask": prepared["attention_mask"],
                "image_mask": prepared["image_mask"],
                "position_embeddings": captured["pos"],
            }
            ours = relevance(layer, state)[0]
        finally:
            handle.remove()

        mask = inputs["attention_mask"]
        length = mask.shape[1]
        allowed = torch.ones(length, length, dtype=torch.bool).tril() & (mask[0][None, :] == 1)
        additive = torch.zeros(1, 1, length, length).masked_fill(
            ~allowed, torch.finfo(torch.float32).min
        )
        attn = layer.self_attn
        previous = attn.config._attn_implementation
        attn.config._attn_implementation = "eager"
        try:
            _, weights = mq.Qwen3VLTextAttention.forward(
                attn,
                layer.input_layernorm(captured["hidden"]),
                captured["pos"],
                attention_mask=additive,
            )
        finally:
            attn.config._attn_implementation = previous
        rows = (mask[0] == 1).nonzero().squeeze(-1)
        image_pos = prepared["image_mask"][0].nonzero().squeeze(-1)
        ref = weights[0].float().mean(0)[rows][:, image_pos].mean(0)
        err = max_rel(ours, ref)
        rank = torch.equal(ours.argsort(), ref.argsort())
        return report(
            f"semantic relevance == layer-{cfg.prune_layer} attention",
            err < TOL and rank,
            f"rel err={err:.3e}, same ranking={rank}",
        )

    # 3 -----------------------------------------------------------------------------------
    def check_action_attention(self, inputs, action_inputs) -> bool:
        backbone = enable_vlapruner(self.model, self.config())
        recorder = backbone.vlap_recorder
        dit = self.model.action_head.model
        calls, handles = [], []
        for i in recorder.blocks:
            attn = dit.transformer_blocks[i].attn1

            def hook(module, args, kwargs, output):
                probs = cross_attention_probs(
                    module, args[0], kwargs["encoder_hidden_states"], kwargs.get("attention_mask")
                )
                value = module.to_v(kwargs["encoder_hidden_states"])
                b, s, _ = value.shape
                value = value.view(b, s, module.heads, -1).transpose(1, 2).float()
                out = (
                    torch.matmul(probs, value)
                    .transpose(1, 2)
                    .reshape(b, -1, value.shape[1] * value.shape[3])
                )
                out = module.to_out[1](module.to_out[0](out.to(output.dtype)))
                act = probs[:, :, -recorder.action_horizon :].mean(dim=(1, 2))
                calls.append((max_rel(out, output), act))

            handles.append(attn.register_forward_hook(hook, with_kwargs=True))
        try:
            out, _ = self.infer(inputs, action_inputs)
        finally:
            for h in handles:
                h.remove()
        recorded = recorder.pop()
        expected = torch.stack([c[1] for c in calls]).mean(0)
        worst = max(c[0] for c in calls)
        err = max_rel(recorded, expected)
        image_cols = out["image_mask"][0]
        n_steps = self.model.action_head.num_inference_timesteps
        return report(
            "DiT action attention == real attention modules",
            worst < TOL and err < 1e-6 and len(calls) == len(recorder.blocks) * n_steps,
            f"output reproduction rel err={worst:.3e}, blocks {recorder.blocks} x {n_steps} steps, "
            f"score mean err={err:.3e}, image-column mass={float(recorded[0][image_cols].sum()):.3f}",
        )

    # 4, 5 --------------------------------------------------------------------------------
    def check_pruned(self, frames) -> bool:
        cfg = self.config(record=True)
        backbone = enable_vlapruner(self.model, cfg)
        for f in frames[: cfg.history_window]:
            self.infer(*f)
        inputs, action_inputs = frames[cfg.history_window]
        out, act = self.infer(inputs, action_inputs)
        rec = backbone.vlap_last
        ok = rec is not None and bool(torch.isfinite(act).all())
        kept = [len(s["kept"]) for s in rec["selections"][0]]
        expected = [max(1, round(n * cfg.keep_ratio)) for n in rec["counts"]]
        ok &= kept == expected
        from_candidates = all(
            set(s["kept"].tolist()) <= set(s["candidates"].tolist()) for s in rec["selections"][0]
        )
        ok &= from_candidates
        filtered = sum(len(s["candidates"]) > len(s["kept"]) for s in rec["selections"][0])

        # Drop == mask: replay the same selection with the dropped tokens only masked out.
        # (run_layers reuses this inference's semantic / action scores and features.)
        prepared = backbone.prepare_pruned_inputs(inputs)
        ref_hidden, ref_state = backbone.run_layers(prepared, emulate_drop=True)
        keep = ref_state["keep"][0]
        valid = out["backbone_attention_mask"][0]
        err = max_rel(out["backbone_features"][0][valid], ref_hidden[0][keep][valid])
        ok &= err < TOL
        ok &= report(
            f"pruned inference (keep={cfg.keep_ratio})",
            ok,
            f"kept per view {kept} of {rec['counts']}, union filtered in {filtered}/{len(kept)} views, "
            f"seq {inputs['input_ids'].shape[1]}->{out['backbone_features'].shape[1]}, "
            f"drop == masked rel err={err:.3e}",
        )

        # History entry of the pruned inference: zeros exactly at the pruned tokens.
        self.infer(*frames[-1])  # flushes the pruned inference's DiT attention into the history
        entry = backbone.vlap_history[-1] if len(frames) > cfg.history_window + 1 else None
        hist_ok = True
        if entry is not None:
            kept_flat = torch.zeros(entry.numel(), dtype=torch.bool)
            offset = 0
            for n, s in zip(rec["counts"], rec["selections"][0]):
                kept_flat[s["kept"] + offset] = True
                offset += n
            hist_ok = bool((entry[~kept_flat] == 0).all()) and bool((entry[kept_flat] > 0).all())
        backbone.reset_history()
        hist_ok &= len(backbone.vlap_history) == 0
        return (
            report(
                "history entries and reset",
                ok and hist_ok,
                "pruned tokens score 0, kept > 0; reset empties the history",
            )
            and ok
        )


def main(args: Args):
    with no_flash_attention():
        policy = Gr00tPolicy(args.embodiment_tag, args.model_path, device="cpu")
    policy.model.to(torch.float32)
    checker = Checker(policy, args.keep_ratio)
    print(f"views={checker.view_keys}, steps={args.steps}")
    observations = load_observations(policy, args.dataset_path, args.traj_id, list(args.steps))
    all_ok = True
    with torch.inference_mode():
        frames = [build_model_inputs(policy, obs, dtype=torch.float32) for obs in observations]
        print(f"sequence {tuple(frames[0][0]['input_ids'].shape)}")
        all_ok &= checker.check_unpruned(frames)
        all_ok &= checker.check_semantic(frames[0][0])
        all_ok &= checker.check_action_attention(*frames[0])
        all_ok &= checker.check_pruned(frames)
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main(tyro.cli(Args))
