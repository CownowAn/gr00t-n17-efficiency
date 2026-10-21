# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLA-Pruner (arXiv 2511.16449) for GR00T N1.7, training-free: semantic relevance from LLM
layer-K attention plus temporally smoothed action relevance from the DiT's action -> image
cross-attention, combined then filtered for redundancy (batch size 1)."""

from .config import VLAPrunerConfig
from .policy import Gr00tVLAPrunerPolicy


__all__ = ["Gr00tVLAPrunerPolicy", "VLAPrunerConfig"]
