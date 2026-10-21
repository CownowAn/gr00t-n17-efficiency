# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field


BUDGET_POLICIES = ("per_view_uniform", "global", "per_view_weighted")
TEXT_SOURCES = ("task", "all_non_image")


@dataclass
class ADPConfig:
    # Fraction of image tokens kept. 1.0 runs the ADP forward path without dropping anything.
    keep_ratio: float = 0.75
    # How the keep budget is split across images (views x frames).
    budget_policy: str = "per_view_uniform"
    # Per-view weights for "per_view_weighted", keyed by video modality key.
    view_weights: dict[str, float] = field(default_factory=dict)
    # Minimum number of tokens kept per image, for every policy.
    min_per_view: int = 1
    # Which text tokens act as queries: the task instruction after the last <|vision_end|>,
    # or every non-image, non-padding token (closer to the original ADP).
    text_source: str = "task"
    # Video modality keys in processor order; set from the policy's modality config.
    view_keys: list[str] = field(default_factory=list)
    # Keep per-sample scores and selections on the backbone for visualization.
    record: bool = False

    def __post_init__(self):
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.budget_policy not in BUDGET_POLICIES:
            raise ValueError(f"budget_policy must be one of {BUDGET_POLICIES}")
        if self.text_source not in TEXT_SOURCES:
            raise ValueError(f"text_source must be one of {TEXT_SOURCES}")
        if self.min_per_view < 0:
            raise ValueError("min_per_view must be >= 0")
