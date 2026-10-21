# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
import json


TEXT_SOURCES = ("task", "all_non_image")


@dataclass
class EfficientVLAConfig:
    # (1) LLM layer pruning: original layer indices to skip (identity). Usually chosen with
    # ``drop_layers_from_importance`` from a ``layer_importance`` calibration file.
    drop_layers: tuple[int, ...] = ()
    # (2) Visual token pruning: fraction of image tokens kept in each view (per_view_uniform).
    # 1.0 disables it.
    keep_ratio: float = 0.5
    # Number of LLM layers run on all tokens; scores come from the attention of the last of
    # them, and tokens are dropped before layer ``prune_after`` (paper: after the 2nd layer).
    prune_after: int = 2
    # Top-scoring tokens always kept in each view (K_key).
    num_key_tokens: int = 4
    # Share of the remaining budget filled by relevance; the rest is filled by diversity.
    alpha: float = 0.5
    # Query text tokens for the relevance score (see gr00t_adp.ADPConfig.text_source).
    text_source: str = "task"
    # (3) DiT feature caching: attention / MLP outputs of every DiT block are recomputed on
    # denoising steps with ``step % cache_interval == 0`` and reused otherwise. 1 disables it.
    cache_interval: int = 2
    # Video modality keys in processor order; set from the policy's modality config.
    view_keys: list[str] = field(default_factory=list)
    # Keep per-sample scores and selections on the backbone for inspection.
    record: bool = False

    def __post_init__(self):
        self.drop_layers = tuple(sorted(set(int(i) for i in self.drop_layers)))
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.prune_after < 1:
            raise ValueError("prune_after must be >= 1")
        if self.num_key_tokens < 0:
            raise ValueError("num_key_tokens must be >= 0")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.text_source not in TEXT_SOURCES:
            raise ValueError(f"text_source must be one of {TEXT_SOURCES}")
        if self.cache_interval < 1:
            raise ValueError("cache_interval must be >= 1")

    @classmethod
    def disabled(cls, **kwargs) -> "EfficientVLAConfig":
        """All three components off: reproduces the original model."""
        return cls(**{"drop_layers": (), "keep_ratio": 1.0, "cache_interval": 1, **kwargs})


def drop_layers_from_importance(path: str, num_drop: int) -> tuple[int, ...]:
    """The ``num_drop`` least important layers of a ``layer_importance`` JSON file."""
    with open(path) as f:
        ranking = json.load(f)["ranking"]
    if num_drop > len(ranking):
        raise ValueError(f"cannot drop {num_drop} of {len(ranking)} layers")
    return tuple(sorted(ranking[:num_drop]))
