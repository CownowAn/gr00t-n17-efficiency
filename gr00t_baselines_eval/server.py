# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy server for the original model or a baseline, for ``gr00t_baselines_eval.rollout``.

Same protocol as ``gr00t/eval/run_gr00t_server.py --use-sim-policy-wrapper``, plus a
``stats`` endpoint: the method, its effective config, and the GPU time of each
``get_action`` call since the last ``stats`` call with ``clear=True``.

    CUDA_VISIBLE_DEVICES=<gpu> python -m gr00t_baselines_eval.server \
        --model-path $CKPT --embodiment-tag LIBERO_PANDA --method vlapruner \
        --config '{"keep_ratio": 0.25}' --port 5555
"""

from dataclasses import dataclass
import json
import time
from typing import Any

from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
from gr00t.policy.policy import PolicyWrapper
from gr00t.policy.server_client import PolicyServer
import numpy as np
import torch
import tyro

from .methods import METHODS, build_policy, describe


@dataclass
class Args:
    model_path: str
    method: str = "original"
    """One of original, adp, efficientvla, vlacache, vlapruner."""
    config: str = "{}"
    """JSON object of config overrides for the method (see gr00t_baselines_eval.methods)."""
    embodiment_tag: str = "LIBERO_PANDA"
    device: str = "cuda"
    host: str = "127.0.0.1"
    port: int = 5555


class TimedPolicy(PolicyWrapper):
    """Forwards to the sim policy wrapper and records the wall time of each call."""

    def __init__(self, policy, method: str, config: dict):
        super().__init__(policy, strict=False)
        self.method = method
        self.config = config
        self.times_ms: list[float] = []

    def get_action(self, observation, options=None):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = self.policy.get_action(observation, options)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times_ms.append((time.perf_counter() - start) * 1e3)
        return out

    # BasePolicy abstract methods; validation is done by the wrapped policy.
    def _get_action(self, observation, options=None):
        return self.policy.get_action(observation, options)

    def check_observation(self, observation):
        pass

    def check_action(self, action):
        pass

    def get_modality_config(self):
        return self.policy.get_modality_config()

    def stats(self, clear: bool = False) -> dict[str, Any]:
        times = np.asarray(self.times_ms)
        out = {
            "method": self.method,
            "config": self.config,
            "n_calls": int(times.size),
            "mean_ms": float(times.mean()) if times.size else None,
            "median_ms": float(np.median(times)) if times.size else None,
        }
        if clear:
            self.times_ms = []
        return out


def main(args: Args):
    if args.method not in METHODS:
        raise ValueError(f"unknown method {args.method!r}; choose from {METHODS}")
    policy = build_policy(
        args.method, args.embodiment_tag, args.model_path, args.device, json.loads(args.config)
    )
    config = json.loads(json.dumps(describe(policy), default=list))
    print(f"method={args.method} config={config}")
    timed = TimedPolicy(Gr00tSimPolicyWrapper(policy), args.method, config)
    with PolicyServer(policy=timed, host=args.host, port=args.port) as server:
        server.register_endpoint("stats", timed.stats)
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    main(tyro.cli(Args))
