# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EfficientVLA (arXiv 2506.10100) for GR00T N1.7, training-free:
LLM layer pruning, task-relevance + diversity visual token pruning, DiT feature caching."""

from .config import EfficientVLAConfig, drop_layers_from_importance
from .policy import Gr00tEfficientVLAPolicy, enable_efficientvla


__all__ = [
    "EfficientVLAConfig",
    "Gr00tEfficientVLAPolicy",
    "drop_layers_from_importance",
    "enable_efficientvla",
]
