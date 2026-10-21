# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 latency and token counts of the original model and each baseline, as in closed loop.

Every run replays the same trajectory of a LeRobot dataset as an episode: one inference every
``--stride`` steps (= n_action_steps), with ``policy.reset()`` at the start, so stateful methods
(VLA-Cache, VLA-Pruner) see consecutive frames. The first ``--skip`` inferences of each episode
(VLA-Pruner's unpruned history fill, VLA-Cache's first full pass) are left out of the summary.

Per inference:
- policy: ``policy.get_action`` end to end (CPU preprocessing + model + decode)
- model: ``model.get_action`` (backbone + action head); backbone timed by forward hooks,
  head = model - backbone
- tokens: image tokens entering the LLM, query tokens each LLM layer computes (0 for a dropped
  layer, measured at the layer's MLP), and the sequence / image tokens the action head reads

    CUDA_VISIBLE_DEVICES=<gpu> python -m gr00t_baselines_eval.benchmark_latency \
        --model-path <ckpt> --dataset-path <dataset> --output-json <out.json> \
        --runs 'original|original|{}' 'vlapruner_0.5|vlapruner|{"keep_ratio": 0.5}'
"""

from dataclasses import dataclass
import gc
import json
import time

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t_adp.benchmark_latency import summarize
from gr00t_adp.data_utils import load_observations
import numpy as np
import torch
import tyro

from .methods import build_policy, describe


@dataclass
class Args:
    model_path: str
    dataset_path: str
    output_json: str
    runs: tuple[str, ...] = ("original|original|{}",)
    """``name|method|JSON config`` per run (config as for ``gr00t_baselines_eval.server``)."""
    embodiment_tag: str = "LIBERO_PANDA"
    traj_ids: tuple[int, ...] = (0, 1)
    stride: int = 8
    max_inferences: int = 40
    """Per episode (trajectory)."""
    skip: int = 3
    warmup_episodes: int = 1
    repeats: int = 2
    """Measured passes over all trajectories."""
    device: str = "cuda"
    seed: int = 0


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Probe:
    """Timing and token-count hooks on a built policy; one record per ``get_action``."""

    def __init__(self, policy):
        self.policy = policy
        backbone = policy.model.backbone
        lm = backbone.model.model.language_model
        self.merge = backbone.model.model.visual.spatial_merge_size
        self.cur: dict = {}
        self.handles = [
            backbone.register_forward_pre_hook(self._backbone_pre, with_kwargs=False),
            backbone.register_forward_hook(self._backbone_post),
        ]
        for i, layer in enumerate(lm.layers):
            self.handles.append(layer.mlp.register_forward_pre_hook(self._layer_hook(i)))
        self.n_layers = len(lm.layers)
        model_get_action = policy.model.get_action

        def timed_model_get_action(*a, **k):
            sync()
            t0 = time.perf_counter()
            out = model_get_action(*a, **k)
            sync()
            self.cur["model_ms"] = (time.perf_counter() - t0) * 1e3
            return out

        policy.model.get_action = timed_model_get_action

    def _backbone_pre(self, _module, args):
        vl_input = args[0]
        sync()
        self.cur["_t0"] = time.perf_counter()
        n_image = int((vl_input["image_grid_thw"].prod(-1) // self.merge**2).sum())
        self.cur.update(seq_in=int(vl_input["input_ids"].shape[1]), image_in=n_image)
        self.cur["layer_tokens"] = [0] * self.n_layers

    def _backbone_post(self, _module, _args, out):
        sync()
        self.cur["backbone_ms"] = (time.perf_counter() - self.cur.pop("_t0")) * 1e3
        self.cur.update(
            head_seq=int(out["backbone_features"].shape[1]),
            head_image=int(out["image_mask"][0].sum()),
        )

    def _layer_hook(self, i):
        def hook(_module, args):
            self.cur["layer_tokens"][i] += int(args[0].shape[-2])

        return hook

    def get_action(self, obs) -> dict:
        self.cur = {}
        sync()
        t0 = time.perf_counter()
        self.policy.get_action(obs)
        sync()
        rec = dict(self.cur, policy_ms=(time.perf_counter() - t0) * 1e3)
        rec["head_ms"] = rec["model_ms"] - rec["backbone_ms"]
        text = rec["seq_in"] - rec["image_in"]
        # Only image tokens are ever pruned / reused, so image queries = queries - text.
        rec["layer_image_tokens"] = [max(n - text, 0) if n else 0 for n in rec["layer_tokens"]]
        return rec

    def remove(self):
        for h in self.handles:
            h.remove()


def run_episodes(probe, episodes, args, measure: bool) -> list[dict]:
    records = []
    for obs_list in episodes:
        probe.policy.reset()
        torch.manual_seed(args.seed)
        for k, obs in enumerate(obs_list):
            with torch.inference_mode():
                rec = probe.get_action(obs)
            if measure and k >= args.skip:
                records.append(rec)
    return records


def summarize_run(records: list[dict]) -> dict:
    out = {
        k: summarize([r[k] for r in records])
        for k in ("policy_ms", "model_ms", "backbone_ms", "head_ms")
    }
    layer_tokens = np.mean([r["layer_tokens"] for r in records], axis=0)
    layer_image = np.mean([r["layer_image_tokens"] for r in records], axis=0)
    out.update(
        n_inferences=len(records),
        seq_in=records[0]["seq_in"],
        image_in=records[0]["image_in"],
        layers_run=float(np.mean([sum(n > 0 for n in r["layer_tokens"]) for r in records])),
        layer_tokens=layer_tokens.tolist(),
        layer_image_tokens=layer_image.tolist(),
        # Token-layers: sum over layers of computed tokens, relative to the original.
        llm_token_layers=float(layer_tokens.sum()),
        llm_image_token_layers=float(layer_image.sum()),
        head_seq=float(np.mean([r["head_seq"] for r in records])),
        head_image=float(np.mean([r["head_image"] for r in records])),
    )
    return out


def parse_run(spec: str) -> tuple[str, str, dict]:
    name, method, config = spec.split("|", 2)
    return name, method, json.loads(config)


def main(args: Args):
    runs = [parse_run(s) for s in args.runs]
    results = {"args": vars(args), "runs": {}}
    episodes = None
    for name, method, config in runs:
        policy = build_policy(method, args.embodiment_tag, args.model_path, args.device, config)
        if episodes is None:
            modality = policy.get_modality_config()
            loader = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=modality)
            episodes = []
            for t in args.traj_ids:
                length = len(loader[t])
                steps = list(range(0, length, args.stride))[: args.max_inferences]
                episodes.append(load_observations(policy, args.dataset_path, t, steps))
            print(f"episodes: {[len(e) for e in episodes]} inferences (stride {args.stride})")
        probe = Probe(policy)
        run_episodes(probe, episodes[: args.warmup_episodes], args, measure=False)
        records = []
        for _ in range(args.repeats):
            records += run_episodes(probe, episodes, args, measure=True)
        summary = summarize_run(records)
        summary.update(method=method, config=json.loads(json.dumps(describe(policy), default=list)))
        results["runs"][name] = summary
        print(
            f"{name}: policy {summary['policy_ms']['mean']:.1f} ms, model {summary['model_ms']['mean']:.1f}, "
            f"backbone {summary['backbone_ms']['mean']:.1f}, head {summary['head_ms']['mean']:.1f}; "
            f"image tokens per layer {np.round(summary['layer_image_tokens']).astype(int).tolist()}, "
            f"head image tokens {summary['head_image']:.1f}"
        )
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        probe.remove()
        del probe, policy
        gc.collect()
        torch.cuda.empty_cache()

    base = results["runs"].get("original")
    print(
        f"\n{'run':<22}{'policy':>8}{'model':>8}{'bbone':>8}{'head':>8}{'speedup':>9}"
        f"{'layers':>8}{'img tok-layers':>16}{'head img':>10}"
    )
    for name, r in results["runs"].items():
        speed = base["model_ms"]["mean"] / r["model_ms"]["mean"] if base else float("nan")
        rel = r["llm_image_token_layers"] / base["llm_image_token_layers"] if base else float("nan")
        print(
            f"{name:<22}{r['policy_ms']['mean']:>8.1f}{r['model_ms']['mean']:>8.1f}"
            f"{r['backbone_ms']['mean']:>8.1f}{r['head_ms']['mean']:>8.1f}{speed:>8.2f}x"
            f"{r['layers_run']:>8.1f}{r['llm_image_token_layers']:>9.0f} ({rel:.0%})"
            f"{r['head_image']:>10.1f}"
        )
    print(f"saved {args.output_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
