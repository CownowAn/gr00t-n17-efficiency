# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latency ablation of EfficientVLA components (in the format of the paper's Table 5).

Rows switch the three components on and off: L = layer pruning (``--num-drop-layers`` least
important layers from ``--importance-json``), T = token pruning (``--keep-ratio``),
C = DiT caching (``--cache-interval``). Per row:

- backbone / head: timed separately with a sync in between
- model: backbone + action head timed as one block (what inference actually costs)
- act dev: relative L2 distance of action_pred from the original model (same noise seed),
  a cheap fidelity signal, not a success rate

    CUDA_VISIBLE_DEVICES=<gpu> python -m gr00t_efficientvla.benchmark_latency \
        --model-path <ckpt> --dataset-path <dataset> --importance-json <layer_importance.json>
"""

from dataclasses import dataclass
import json

from gr00t_adp.benchmark_latency import Timer, summarize
from gr00t_adp.data_utils import build_model_inputs, load_observations, stack_observations
import torch
from transformers.feature_extraction_utils import BatchFeature
import tyro

from .config import EfficientVLAConfig, drop_layers_from_importance
from .policy import Gr00tEfficientVLAPolicy


@dataclass
class Args:
    model_path: str
    dataset_path: str
    importance_json: str
    embodiment_tag: str = "new_embodiment"
    traj_id: int = 0
    step: int = 40
    batch_size: int = 1
    num_drop_layers: int = 4
    keep_ratio: float = 0.5
    cache_interval: int = 2
    device: str = "cuda"
    warmup: int = 10
    iters: int = 50
    output_json: str | None = None


ROWS = [  # (L, T, C)
    (False, False, False),
    (False, True, False),
    (False, False, True),
    (True, False, False),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
]


def row_name(layers: bool, tokens: bool, cache: bool) -> str:
    parts = [n for n, on in zip("LTC", (layers, tokens, cache)) if on]
    return "+".join(parts) if parts else "original"


def bench_row(policy, config, inputs, action_inputs, args, timer):
    policy.set_config(config)
    backbone, head = policy.backbone, policy.model.action_head

    def model():
        torch.manual_seed(0)
        return head.get_action(backbone(inputs), action_inputs)["action_pred"]

    times = {"backbone": [], "head": [], "model": []}
    for i in range(args.warmup + args.iters):
        action, t_model = timer(model)
        out, t_bb = timer(lambda: backbone(inputs))
        torch.manual_seed(0)
        # The action head overwrites backbone_features in place; give it a copy.
        _, t_head = timer(lambda: head.get_action(BatchFeature(data=dict(out)), action_inputs))
        if i >= args.warmup:
            for k, v in zip(times, (t_bb, t_head, t_model)):
                times[k].append(v)
    result = {k: summarize(v) for k, v in times.items()}
    result.update(
        seq_len=out["backbone_features"].shape[1], image_tokens=int(out["image_mask"][0].sum())
    )
    return result, action.float()


def main(args: Args):
    drop = drop_layers_from_importance(args.importance_json, args.num_drop_layers)
    timer = Timer(torch.device(args.device))
    policy = Gr00tEfficientVLAPolicy(
        args.embodiment_tag,
        args.model_path,
        device=args.device,
        evla_config=EfficientVLAConfig.disabled(),
    )
    policy.model.to(torch.bfloat16)
    attn_impl = policy.backbone.model.config._attn_implementation
    print(f"attn={attn_impl}, views={policy.evla_config.view_keys}, drop_layers={drop}")

    obs = load_observations(policy, args.dataset_path, args.traj_id, [args.step])[0]
    obs = stack_observations([obs] * args.batch_size)
    inputs, action_inputs = build_model_inputs(policy, obs, dtype=torch.bfloat16)
    print(f"batch={args.batch_size}, full sequence={tuple(inputs['input_ids'].shape)}")

    rows, reference = {}, None
    with torch.inference_mode():
        for layers, tokens, cache in ROWS:
            config = EfficientVLAConfig(
                drop_layers=drop if layers else (),
                keep_ratio=args.keep_ratio if tokens else 1.0,
                cache_interval=args.cache_interval if cache else 1,
            )
            result, action = bench_row(policy, config, inputs, action_inputs, args, timer)
            if reference is None:
                reference = action
            result["action_dev"] = float((action - reference).norm() / reference.norm())
            rows[row_name(layers, tokens, cache)] = result

    base = rows["original"]["model"]["mean"]
    print(
        f"\nmean latency in ms ({args.iters} iters, warmup {args.warmup}); L: drop {drop}, "
        f"T: keep {args.keep_ratio}, C: interval {args.cache_interval}"
    )
    print(
        f"{'config':<10}{'img tok':>8}{'seq':>6}{'backbone':>10}{'head':>9}{'model':>9}"
        f"{'speedup':>9}{'act dev':>9}"
    )
    for name, r in rows.items():
        print(
            f"{name:<10}{r['image_tokens']:>8}{r['seq_len']:>6}{r['backbone']['mean']:>10.2f}"
            f"{r['head']['mean']:>9.2f}{r['model']['mean']:>9.2f}{base / r['model']['mean']:>8.2f}x"
            f"{r['action_dev']:>9.3f}"
        )
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"args": vars(args), "drop_layers": drop, "rows": rows}, f, indent=2)
        print(f"saved {args.output_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
