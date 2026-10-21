# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LIBERO success rate of a policy served by ``gr00t_baselines_eval.server``, one env at a time.

Differences from ``gr00t/eval/rollout_policy.py`` (whose env / wrapper code is reused):
- one environment (batch size 1, as required by VLA-Cache and VLA-Pruner);
- ``policy.reset()`` after every finished episode, so per-episode state starts empty;
- no videos unless ``--video-dir`` is given;
- every task of a suite in one run, the same seed per task for every method, results written
  after each task (a rerun with the same ``--output-json`` skips finished tasks).

Run with the LIBERO venv (see examples/LIBERO/README.md):

    gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python -m gr00t_baselines_eval.rollout \
        --suite libero_10 --n-episodes 50 --port 5555 --output-json <out>/vlapruner.json
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time

from gr00t.eval import rollout_policy
from gr00t.eval._horizon_contract import PolicyHorizonSpec
from gr00t.eval.rollout_policy import (
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    run_rollout_gymnasium_policy,
)
from gr00t.policy.server_client import PolicyClient
from gr00t.utils.determinism import seed_everything
import numpy as np
import tyro


@dataclass
class Args:
    output_json: str
    suite: str = "libero_10"
    """libero_10, libero_spatial, libero_object or libero_goal."""
    tasks: tuple[int, ...] = ()
    """Task indices within the suite; empty = all."""
    n_episodes: int = 50
    max_episode_steps: int = 720
    n_action_steps: int = 8
    seed: int = 0
    host: str = "127.0.0.1"
    port: int = 5555
    timeout_ms: int = 120000
    video_dir: str | None = None


class _ResetPolicyOnDone:
    """Single-env vector env that resets the policy whenever its episode ends (autoreset)."""

    def __init__(self, env, policy):
        self._env = env
        self._policy = policy

    def step(self, actions):
        out = self._env.step(actions)
        terminations, truncations = out[2], out[3]
        if terminations[0] or truncations[0]:
            self._policy.reset()
        return out

    def __getattr__(self, name):
        return getattr(self._env, name)


_collect = rollout_policy._collect_rollout_episodes


def _collect_with_reset(env, policy, n_episodes, n_envs, seed):
    assert n_envs == 1, "per-episode policy reset needs a single env"
    return _collect(_ResetPolicyOnDone(env, policy), policy, n_episodes, n_envs, seed)


def suite_task_names(suite: str) -> list[str]:
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[suite]().get_task_names()


def main(args: Args):
    rollout_policy._collect_rollout_episodes = _collect_with_reset
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {"tasks": {}}

    names = suite_task_names(args.suite)
    indices = list(args.tasks) or list(range(len(names)))
    policy = PolicyClient(host=args.host, port=args.port, timeout_ms=args.timeout_ms)
    server = policy.call_endpoint("stats", {"clear": True})
    previous = results.get("server")
    if previous is not None and (previous["method"], previous["config"]) != (
        server["method"],
        server["config"],
    ):
        raise ValueError(f"{out_path} was written for another server: {previous}")
    results["server"] = server
    results.update(suite=args.suite, n_episodes=args.n_episodes, seed=args.seed)
    print(f"server: {results['server']['method']} {results['server']['config']}")

    for i in indices:
        env_name = f"libero_sim/{names[i]}"
        if env_name in results["tasks"]:
            print(f"skip {env_name} (done)")
            continue
        seed = seed_everything(args.seed)
        contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=args.n_action_steps)
        wrapper_configs = WrapperConfigs(
            multistep=MultiStepConfig(
                contract=contract,
                max_episode_steps=args.max_episode_steps,
                terminate_on_success=True,
            ),
            video=VideoConfig(
                video_dir=None if args.video_dir is None else f"{args.video_dir}/{names[i]}",
                max_episode_steps=args.max_episode_steps,
            ),
        )
        start = time.time()
        _, successes, infos = run_rollout_gymnasium_policy(
            env_name=env_name,
            policy=policy,
            wrapper_configs=wrapper_configs,
            n_episodes=args.n_episodes,
            n_envs=1,
            seed=seed,
        )
        stats = policy.call_endpoint("stats", {"clear": True})
        results["tasks"][env_name] = {
            "successes": [bool(s) for s in successes],
            "success_rate": float(np.mean(successes)),
            "episode_lengths": [int(x) for x in infos["episode_lengths"]],
            "latency_mean_ms": stats["mean_ms"],
            "latency_median_ms": stats["median_ms"],
            "n_inferences": stats["n_calls"],
            "wall_s": time.time() - start,
        }
        done = [t["successes"] for t in results["tasks"].values()]
        results["success_rate"] = float(np.mean([s for t in done for s in t]))
        out_path.write_text(json.dumps(results, indent=2))
        print(
            f"{env_name}: {np.mean(successes):.3f} ({len(successes)} eps), "
            f"latency {stats['median_ms']:.1f} ms; running mean {results['success_rate']:.3f}"
        )
    print(f"suite {args.suite} success rate: {results.get('success_rate')}")


if __name__ == "__main__":
    main(tyro.cli(Args))
