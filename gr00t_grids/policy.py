# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from safetensors import safe_open
import torch

from .backbone import build_samplers, enable_grids
from .config import GridSConfig


SAMPLER_PREFIX = "backbone.grid_sampler."


def load_sampler_state(model_path: str | Path) -> dict[str, torch.Tensor]:
    """Grid Sampler weights from a checkpoint's safetensors (sharded or not)."""
    model_path = Path(model_path)
    index = model_path / "model.safetensors.index.json"
    files = (
        sorted({model_path / f for f in json.loads(index.read_text())["weight_map"].values()})
        if index.exists()
        else [model_path / "model.safetensors"]
    )
    state = {}
    for file in files:
        with safe_open(str(file), framework="pt") as f:
            for key in f.keys():
                if key.startswith(SAMPLER_PREFIX):
                    state[key[len(SAMPLER_PREFIX) :]] = f.get_tensor(key)
    if not state:
        raise KeyError(f"no '{SAMPLER_PREFIX}*' weights in {model_path}")
    return state


class Gr00tGridSPolicy(Gr00tPolicy):
    """``Gr00tPolicy`` for checkpoints trained with ``gr00t_grids/launch_finetune.py``."""

    def __init__(
        self,
        embodiment_tag: EmbodimentTag | str,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
    ):
        # The base loader ignores the (unexpected) sampler weights; they are attached below.
        super().__init__(embodiment_tag, model_path, device=device, strict=strict)
        self.grids_config = GridSConfig.load(Path(model_path) / "experiment_cfg")
        backbone = self.model.backbone
        vision_dim = backbone.model.config.text_config.hidden_size
        samplers = build_samplers(self.grids_config, vision_dim)
        samplers.load_state_dict(load_sampler_state(model_path))
        param = next(backbone.parameters())
        enable_grids(
            backbone, self.grids_config, samplers.to(device=param.device, dtype=param.dtype)
        )
        self.model.eval()

    @property
    def backbone(self):
        return self.model.backbone
