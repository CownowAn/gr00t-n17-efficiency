# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Action-aware Dynamic Pruning (ADP) for GR00T N1.7, implemented without modifying ``gr00t/``.

Only the text-driven visual-token selection is implemented here (static pruning). The
action-aware on/off gate from ADP is a later step.
"""

from .config import ADPConfig
from .qwen3_backbone_adp import Qwen3BackboneADP, enable_adp


__all__ = ["ADPConfig", "Qwen3BackboneADP", "enable_adp"]
