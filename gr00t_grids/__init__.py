# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differentiable Grid Sampler (GridS, arXiv 2605.11817) for GR00T N1.7: a learned sampler
reads K tokens per image at predicted continuous coordinates before the LLM. Trained with
``gr00t_grids/launch_finetune.py`` (same arguments as the original launcher)."""

from .config import GridSConfig
from .policy import Gr00tGridSPolicy


__all__ = ["GridSConfig", "Gr00tGridSPolicy"]
