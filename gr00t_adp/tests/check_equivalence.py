# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical checks of the ADP backbone against the original GR00T backbone on real data.

1. Scores: ``text_to_image_scores`` equals the logits of the real layer-0 attention module
   (eager, identity RoPE) up to a per-row constant.
2. keep_ratio=1.0: the unrolled ADP forward reproduces the original backbone features
   and the action head output.
3. keep_ratio<1: dropping tokens equals running the full sequence with the dropped tokens
   masked out as keys (checks the gather of embeds / MRoPE ids / DeepStack features).
4. Per-view budgets of every policy.

Run (CPU, fp32):
    CUDA_VISIBLE_DEVICES="" python -m gr00t_adp.tests.check_equivalence --model-path <ckpt> \
        --dataset-path <lerobot dataset>
"""

from dataclasses import dataclass
import time

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
import torch
import tyro

from gr00t_adp import ADPConfig, enable_adp
from gr00t_adp.adp_policy import no_flash_attention
from gr00t_adp.data_utils import build_model_inputs, load_observations, stack_observations
from gr00t_adp.qwen3_backbone_adp import disable_adp
from gr00t_adp.token_scoring import text_to_image_scores


@dataclass
class Args:
    model_path: str
    dataset_path: str
    embodiment_tag: str = "new_embodiment"
    traj_id: int = 0
    steps: tuple[int, ...] = (0, 40)
    keep_ratio: float = 0.5
    device: str = "cpu"
    dtype: str = "float32"


def report(name: str, ok: bool, detail: str):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def check_scores(backbone, inputs, tol) -> bool:
    """Compare against the real attention module run on [image; text] with identity RoPE."""
    from transformers.models.qwen3_vl import modeling_qwen3_vl as mq

    prepared = backbone.prepare_pruned_inputs(inputs, apply_drop=False)
    layer = backbone.model.model.language_model.layers[0]
    ids, mask, image_mask = (
        inputs["input_ids"][0],
        prepared["attention_mask"][0],
        prepared["image_mask"][0],
    )
    image_pos = image_mask.nonzero().squeeze(-1)
    text_pos = backbone._text_positions(ids, mask, image_mask)
    hidden = prepared["inputs_embeds"][0]
    ours = text_to_image_scores(layer, hidden, text_pos, image_pos)

    attn = layer.self_attn
    x = layer.input_layernorm(hidden[torch.cat([image_pos, text_pos])])[None]
    head_dim = attn.head_dim
    cos = torch.ones(1, x.shape[1], head_dim, dtype=x.dtype, device=x.device)
    sin = torch.zeros_like(cos)
    previous = attn.config._attn_implementation
    attn.config._attn_implementation = "eager"
    try:
        _, weights = mq.Qwen3VLTextAttention.forward(attn, x, (cos, sin), attention_mask=None)
    finally:
        attn.config._attn_implementation = previous
    n_img = image_pos.numel()
    # log softmax = logit - logsumexp(row); averaging rows leaves one constant over keys.
    ref = weights[0, :, n_img:, :n_img].float().clamp_min(1e-30).log().mean(dim=(0, 1))
    diff = ours - ref
    spread = float(diff.max() - diff.min())
    rank_ok = torch.equal(ours.argsort(), ref.argsort())
    return report(
        "scores vs attention module",
        spread < tol * float(ours.abs().max()) and rank_ok,
        f"offset spread={spread:.3e}, |scores|max={float(ours.abs().max()):.3f}, same ranking={rank_ok}",
    )


def check_ratio_one(policy, backbone, inputs, action_inputs, tol) -> bool:
    disable_adp(backbone)
    ref = Qwen3Backbone.forward(backbone, inputs)
    enable_adp(backbone, ADPConfig(keep_ratio=1.0, view_keys=backbone_view_keys(policy)))
    out = backbone(inputs)
    ok = torch.equal(ref["backbone_attention_mask"], out["backbone_attention_mask"])
    ok &= torch.equal(ref["image_mask"], out["image_mask"])
    valid = ref["backbone_attention_mask"]
    err = max_rel(out["backbone_features"][valid], ref["backbone_features"][valid])
    ok = report("ratio=1.0 backbone features", ok and err < tol, f"max rel err={err:.3e}")

    head = policy.model.action_head
    actions = []
    for o in (ref, out):
        torch.manual_seed(0)
        actions.append(head.get_action(o, action_inputs)["action_pred"])
    err = max_rel(actions[1], actions[0])
    return report("ratio=1.0 action_pred", err < tol, f"max rel err={err:.3e}") and ok


def check_drop(backbone, inputs, keep_ratio, tol) -> bool:
    backbone.adp_config = ADPConfig(keep_ratio=keep_ratio, view_keys=backbone.adp_config.view_keys)
    full = backbone.prepare_pruned_inputs(inputs, apply_drop=False)
    keep = full["keep_mask"]
    masked = full["attention_mask"] * keep.to(full["attention_mask"].dtype)
    ref = backbone.run_language_model(full, attention_mask=masked)
    out = backbone(inputs)
    feats = out["backbone_features"]

    ok = True
    worst = 0.0
    for b in range(feats.shape[0]):
        ref_b = ref[b][keep[b]]
        valid = out["backbone_attention_mask"][b]
        ok &= ref_b.shape == feats[b].shape
        worst = max(worst, max_rel(feats[b][valid], ref_b[valid]))
    n_img_before = int(full["image_mask"][0].sum())
    n_img_after = int(out["image_mask"][0].sum())
    return report(
        f"drop == masked full sequence (keep_ratio={keep_ratio})",
        ok and worst < tol,
        f"image tokens {n_img_before}->{n_img_after}, seq {keep.shape[1]}->{feats.shape[1]}, "
        f"max rel err={worst:.3e}",
    )


def check_budgets(backbone, inputs, keep_ratio) -> bool:
    keys = backbone.adp_config.view_keys
    weights = {k: float(len(keys) - i) for i, k in enumerate(keys)}  # descending by view order
    ok = True
    for policy, kwargs in [
        ("per_view_uniform", {}),
        ("global", {}),
        ("per_view_weighted", {"view_weights": weights}),
    ]:
        backbone.adp_config = ADPConfig(
            keep_ratio=keep_ratio, budget_policy=policy, view_keys=keys, record=True, **kwargs
        )
        out = backbone(inputs)
        rec = backbone.adp_last
        kept = [[len(k) for k in sample] for sample in rec["kept"]]
        total, n_kept = sum(rec["counts"]), sum(kept[0])
        n_img_out = int(out["image_mask"][0].sum())
        # Per-image counts may differ across samples for "global"; the total may not.
        good = n_kept == n_img_out and all(sum(k) == n_kept for k in kept)
        good &= all(k >= 1 for sample in kept for k in sample)
        if policy != "global":
            good &= all(k == kept[0] for k in kept)
        if policy != "per_view_uniform":
            good &= n_kept == max(len(keys), round(total * keep_ratio))
        ok &= report(f"budget {policy}", good, f"kept per image {kept[0]} of {rec['counts']}")
    return ok


def backbone_view_keys(policy) -> list[str]:
    return list(policy.modality_configs["video"].modality_keys)


def main(args: Args):
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    dtype = getattr(torch, args.dtype)
    tol = 1e-4 if dtype == torch.float32 else 3e-2
    t0 = time.time()
    with no_flash_attention():
        policy = Gr00tPolicy(args.embodiment_tag, args.model_path, device=args.device)
    policy.model.to(dtype)
    print(f"loaded in {time.time() - t0:.1f}s; views={backbone_view_keys(policy)}")

    observations = load_observations(policy, args.dataset_path, args.traj_id, list(args.steps))
    # Second sample gets a longer instruction so the batch has left padding.
    second = observations[1]
    lang_key = next(iter(second["language"]))
    second["language"][lang_key] = [[second["language"][lang_key][0][0] + " carefully and quickly"]]
    batches = {"B=1": observations[0], "B=2 (padded)": stack_observations(observations[:2])}

    backbone = policy.model.backbone
    enable_adp(backbone, ADPConfig(keep_ratio=1.0, view_keys=backbone_view_keys(policy)))
    all_ok = True
    with torch.inference_mode():
        for name, obs in batches.items():
            print(f"--- {name}")
            inputs, action_inputs = build_model_inputs(policy, obs, dtype=dtype)
            print(
                f"sequence length {inputs['input_ids'].shape}, images {inputs['image_grid_thw'].shape[0]}"
            )
            all_ok &= check_scores(backbone, inputs, tol)
            all_ok &= check_ratio_one(policy, backbone, inputs, action_inputs, tol)
            all_ok &= check_drop(backbone, inputs, args.keep_ratio, tol)
            all_ok &= check_budgets(backbone, inputs, args.keep_ratio)
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main(tyro.cli(Args))
