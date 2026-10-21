# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference latency of ADP static pruning per keep ratio (layer-0 scoring).

Rows: ``original`` is the unmodified ``Qwen3Backbone.forward``; ``keep=r`` runs the ADP
backbone. Stages are timed with synchronization between them:

- prep: vision encoder + embedding + (scoring, top-k, gather when r < 1)
- llm: truncated Qwen3-VL language model on the (pruned) sequence
- head: action head (vl_self_attention + DiT denoising loop)
- model: ``prep + llm + head`` timed as one block, without inner syncs
- policy: ``policy.get_action`` end to end (CPU preprocessing + model + decode), optional

    CUDA_VISIBLE_DEVICES=<id> python -m gr00t_adp.benchmark_latency --model-path <ckpt> \
        --dataset-path <dataset> --keep-ratios 1.0 0.75 0.5 0.25
"""

from dataclasses import dataclass
import json
import time

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
import tyro

from gr00t_adp import ADPConfig
from gr00t_adp.adp_policy import Gr00tADPPolicy, no_flash_attention
from gr00t_adp.data_utils import build_model_inputs, load_observations, stack_observations


@dataclass
class Args:
    model_path: str
    dataset_path: str
    embodiment_tag: str = "new_embodiment"
    traj_id: int = 0
    step: int = 40
    batch_size: int = 1
    keep_ratios: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
    budget_policy: str = "per_view_uniform"
    device: str = "cuda"
    warmup: int = 10
    iters: int = 50
    policy_iters: int = 20
    """Iterations of the end-to-end ``policy.get_action``; 0 skips it."""
    output_json: str | None = None


class Timer:
    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def __call__(self, fn):
        self.sync()
        t0 = time.perf_counter()
        out = fn()
        self.sync()
        return out, (time.perf_counter() - t0) * 1e3


def summarize(values: list[float]) -> dict:
    a = np.asarray(values)
    return {"mean": float(a.mean()), "p50": float(np.median(a)), "p90": float(np.percentile(a, 90))}


def bench_model(policy, inputs, action_inputs, keep_ratio, args, timer) -> dict:
    """keep_ratio=None runs the original backbone forward."""
    backbone, head = policy.backbone, policy.model.action_head
    if keep_ratio is not None:
        backbone.adp_config = ADPConfig(
            keep_ratio=keep_ratio,
            budget_policy=args.budget_policy,
            view_keys=policy.adp_config.view_keys,
        )

    def run_backbone():
        if keep_ratio is None:
            return Qwen3Backbone.forward(backbone, inputs)
        return backbone(inputs)

    stages = {"prep": [], "llm": [], "head": [], "model": []}
    seq_len = n_img = None
    for i in range(args.warmup + args.iters):
        torch.manual_seed(0)
        _, t_model = timer(lambda: head.get_action(run_backbone(), action_inputs))
        if keep_ratio is None:
            out, t_bb = timer(run_backbone)
            t_prep = t_llm = float("nan")
        else:
            prepared, t_prep = timer(lambda: backbone.prepare_pruned_inputs(inputs))
            hidden, t_llm = timer(lambda: backbone.run_language_model(prepared))
            out = BatchFeature(
                data={
                    "backbone_features": hidden,
                    "backbone_attention_mask": prepared["attention_mask"] == 1,
                    "image_mask": prepared["image_mask"],
                }
            )
        _, t_head = timer(lambda: head.get_action(out, action_inputs))
        if i >= args.warmup:
            for k, v in zip(stages, (t_prep, t_llm, t_head, t_model)):
                stages[k].append(v)
        seq_len = out["backbone_features"].shape[1]
        n_img = int(out["image_mask"][0].sum())
    result = {k: summarize(v) for k, v in stages.items()}
    result.update(seq_len=seq_len, image_tokens=n_img)
    return result


def bench_policy(policy, obs, keep_ratio, args, timer) -> dict:
    policy.backbone.adp_config = ADPConfig(
        keep_ratio=keep_ratio,
        budget_policy=args.budget_policy,
        view_keys=policy.adp_config.view_keys,
    )
    times = []
    for i in range(args.warmup + args.policy_iters):
        _, t = timer(lambda: policy.get_action(obs))
        if i >= args.warmup:
            times.append(t)
    return summarize(times)


def main(args: Args):
    device = torch.device(args.device)
    timer = Timer(device)
    config = ADPConfig(keep_ratio=1.0, budget_policy=args.budget_policy)
    if device.type == "cpu":
        with no_flash_attention():
            policy = Gr00tADPPolicy(
                args.embodiment_tag, args.model_path, device=args.device, adp_config=config
            )
    else:
        policy = Gr00tADPPolicy(
            args.embodiment_tag, args.model_path, device=args.device, adp_config=config
        )
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    policy.model.to(dtype)
    attn_impl = policy.backbone.model.config._attn_implementation
    print(f"device={device}, dtype={dtype}, attn={attn_impl}, views={policy.adp_config.view_keys}")

    obs = load_observations(policy, args.dataset_path, args.traj_id, [args.step])[0]
    obs = stack_observations([obs] * args.batch_size)
    inputs, action_inputs = build_model_inputs(policy, obs, dtype=dtype)
    print(f"batch={args.batch_size}, full sequence={tuple(inputs['input_ids'].shape)}")

    rows = {}
    with torch.inference_mode():
        rows["original"] = bench_model(policy, inputs, action_inputs, None, args, timer)
        for r in args.keep_ratios:
            rows[f"keep={r}"] = bench_model(policy, inputs, action_inputs, r, args, timer)
    if args.policy_iters > 0:
        rows["original"]["policy"] = None
        for r in args.keep_ratios:
            rows[f"keep={r}"]["policy"] = bench_policy(policy, obs, r, args, timer)

    base = rows["original"]["model"]["mean"]
    header = f"{'config':<12}{'img tok':>8}{'seq':>6}{'prep':>9}{'llm':>9}{'head':>9}{'model':>9}{'speedup':>9}{'policy':>9}"
    print(f"\nmean latency in ms ({args.iters} iters, warmup {args.warmup})\n{header}")
    for name, r in rows.items():
        policy_ms = r.get("policy")
        print(
            f"{name:<12}{r['image_tokens']:>8}{r['seq_len']:>6}"
            f"{r['prep']['mean']:>9.2f}{r['llm']['mean']:>9.2f}{r['head']['mean']:>9.2f}"
            f"{r['model']['mean']:>9.2f}{base / r['model']['mean']:>8.2f}x"
            f"{policy_ms['mean'] if policy_ms else float('nan'):>9.2f}"
        )
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"args": vars(args), "rows": rows}, f, indent=2)
        print(f"saved {args.output_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
