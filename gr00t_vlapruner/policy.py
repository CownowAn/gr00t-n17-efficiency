# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

from .backbone import enable_vlapruner
from .config import VLAPrunerConfig


class Gr00tVLAPrunerPolicy(Gr00tPolicy):
    """``Gr00tPolicy`` with VLA-Pruner (batch size 1).

    The action-relevance history is per episode: call ``reset()`` at every episode start.
    The first ``history_window`` inferences of an episode run unpruned.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag | str,
        model_path: str,
        *,
        device: int | str,
        vlap_config: VLAPrunerConfig,
        strict: bool = True,
    ):
        super().__init__(embodiment_tag, model_path, device=device, strict=strict)
        self.set_config(vlap_config)

    def set_config(self, config: VLAPrunerConfig) -> None:
        self.vlap_config = replace(
            config, view_keys=list(self.modality_configs["video"].modality_keys)
        )
        enable_vlapruner(self.model, self.vlap_config)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self.model.backbone.reset_history()
        return super().reset(options)

    @property
    def backbone(self):
        return self.model.backbone
