# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy import gr00t_policy as gr00t_policy_module
from gr00t.policy.gr00t_policy import Gr00tPolicy

from .config import ADPConfig
from .qwen3_backbone_adp import enable_adp


class Gr00tADPPolicy(Gr00tPolicy):
    """``Gr00tPolicy`` whose backbone prunes image tokens with ADP before the LLM."""

    def __init__(
        self,
        embodiment_tag: EmbodimentTag | str,
        model_path: str,
        *,
        device: int | str,
        adp_config: ADPConfig,
        strict: bool = True,
    ):
        super().__init__(embodiment_tag, model_path, device=device, strict=strict)
        view_keys = list(self.modality_configs["video"].modality_keys)
        unknown = set(adp_config.view_weights) - set(view_keys)
        if unknown:
            raise ValueError(
                f"view_weights has unknown views {sorted(unknown)}; views: {view_keys}"
            )
        self.adp_config = replace(adp_config, view_keys=view_keys)
        enable_adp(self.model.backbone, self.adp_config)

    @property
    def backbone(self):
        return self.model.backbone


@contextmanager
def no_flash_attention():
    """Load checkpoints with SDPA instead of FlashAttention-2 (needed on CPU)."""
    original = gr00t_policy_module.AutoModel.from_pretrained

    def from_pretrained(*args, **kwargs):
        kwargs.setdefault("use_flash_attention", False)
        return original(*args, **kwargs)

    with patch.object(gr00t_policy_module.AutoModel, "from_pretrained", from_pretrained):
        yield
