# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

from .backbone import enable_vlacache_backbone
from .config import VLACacheConfig


class Gr00tVLACachePolicy(Gr00tPolicy):
    """``Gr00tPolicy`` with VLA-Cache token reuse across inferences (batch size 1).

    The cache is per episode: call ``reset()`` at every episode start. It is also rebuilt from
    scratch whenever the prompt or image layout changes.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag | str,
        model_path: str,
        *,
        device: int | str,
        vlac_config: VLACacheConfig,
        strict: bool = True,
    ):
        super().__init__(embodiment_tag, model_path, device=device, strict=strict)
        image_processor = self.processor.processor.image_processor
        self.vlac_config = replace(
            vlac_config,
            view_keys=list(self.modality_configs["video"].modality_keys),
            image_mean=tuple(image_processor.image_mean),
            image_std=tuple(image_processor.image_std),
        )
        enable_vlacache_backbone(self.model.backbone, self.vlac_config)

    def set_config(self, config: VLACacheConfig) -> None:
        self.vlac_config = replace(
            config,
            view_keys=self.vlac_config.view_keys,
            image_mean=self.vlac_config.image_mean,
            image_std=self.vlac_config.image_std,
        )
        enable_vlacache_backbone(self.model.backbone, self.vlac_config)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self.model.backbone.reset_cache()
        return super().reset(options)

    @property
    def backbone(self):
        return self.model.backbone
