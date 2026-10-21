# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the original GR00T N1.7 policy or one of the training-free baselines by name."""

from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


METHODS = ("original", "adp", "efficientvla", "vlacache", "vlapruner")
# Methods that keep state across the inferences of an episode (batch size 1, reset per episode).
STATEFUL = ("vlacache", "vlapruner")


def build_policy(
    method: str,
    embodiment_tag: EmbodimentTag | str,
    model_path: str,
    device: int | str,
    overrides: dict[str, Any] | None = None,
) -> Gr00tPolicy:
    """``overrides`` are fields of the method's config dataclass. For EfficientVLA,
    ``importance_json`` + ``num_drop_layers`` may be given instead of ``drop_layers``."""
    overrides = dict(overrides or {})
    if method == "original":
        if overrides:
            raise ValueError(f"original takes no config, got {overrides}")
        return Gr00tPolicy(embodiment_tag, model_path, device=device)
    if method == "adp":
        from gr00t_adp.adp_policy import Gr00tADPPolicy
        from gr00t_adp.config import ADPConfig

        return Gr00tADPPolicy(
            embodiment_tag, model_path, device=device, adp_config=ADPConfig(**overrides)
        )
    if method == "efficientvla":
        from gr00t_efficientvla import (
            EfficientVLAConfig,
            Gr00tEfficientVLAPolicy,
            drop_layers_from_importance,
        )

        importance = overrides.pop("importance_json", None)
        num_drop = overrides.pop("num_drop_layers", None)
        if (importance is None) != (num_drop is None):
            raise ValueError("give importance_json and num_drop_layers together")
        if importance is not None:
            overrides["drop_layers"] = drop_layers_from_importance(importance, num_drop)
        return Gr00tEfficientVLAPolicy(
            embodiment_tag, model_path, device=device, evla_config=EfficientVLAConfig(**overrides)
        )
    if method == "vlacache":
        from gr00t_vlacache import Gr00tVLACachePolicy, VLACacheConfig

        return Gr00tVLACachePolicy(
            embodiment_tag, model_path, device=device, vlac_config=VLACacheConfig(**overrides)
        )
    if method == "vlapruner":
        from gr00t_vlapruner import Gr00tVLAPrunerPolicy, VLAPrunerConfig

        return Gr00tVLAPrunerPolicy(
            embodiment_tag, model_path, device=device, vlap_config=VLAPrunerConfig(**overrides)
        )
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def describe(policy: Gr00tPolicy) -> dict[str, Any]:
    """The effective config of a built policy (for the results file)."""
    for attr in ("adp_config", "evla_config", "vlac_config", "vlap_config"):
        cfg = getattr(policy, attr, None)
        if cfg is not None:
            return {k: v for k, v in vars(cfg).items() if k not in ("record",)}
    return {}
