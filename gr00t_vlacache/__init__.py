# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLA-Cache (arXiv 2502.02175) for GR00T N1.7, training-free: reuse the KV and last-layer
states of static, task-irrelevant image tokens across inferences (batch size 1)."""

from .config import VLACacheConfig
from .policy import Gr00tVLACachePolicy


__all__ = ["Gr00tVLACachePolicy", "VLACacheConfig"]
