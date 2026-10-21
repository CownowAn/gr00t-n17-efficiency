# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

from .backbone import enable_efficientvla_backbone
from .config import EfficientVLAConfig
from .dit_cache import enable_dit_cache


def enable_efficientvla(model, config: EfficientVLAConfig) -> None:
    """Apply all three EfficientVLA components to a loaded GR00T N1.7 model in place."""
    enable_efficientvla_backbone(model.backbone, config)
    enable_dit_cache(model.action_head, config.cache_interval)


class Gr00tEfficientVLAPolicy(Gr00tPolicy):
    """``Gr00tPolicy`` with EfficientVLA layer pruning, token pruning and DiT caching."""

    def __init__(
        self,
        embodiment_tag: EmbodimentTag | str,
        model_path: str,
        *,
        device: int | str,
        evla_config: EfficientVLAConfig,
        strict: bool = True,
    ):
        super().__init__(embodiment_tag, model_path, device=device, strict=strict)
        self.evla_config = replace(
            evla_config, view_keys=list(self.modality_configs["video"].modality_keys)
        )
        enable_efficientvla(self.model, self.evla_config)

    def set_config(self, config: EfficientVLAConfig) -> None:
        self.evla_config = replace(config, view_keys=self.evla_config.view_keys)
        enable_efficientvla(self.model, self.evla_config)

    @property
    def backbone(self):
        return self.model.backbone
