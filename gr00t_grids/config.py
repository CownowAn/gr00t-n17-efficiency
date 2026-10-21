# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict, dataclass
import json
from pathlib import Path


POSITION_MODES = ("continuous", "grid", "first")
CONFIG_FILENAME = "grids_config.json"


@dataclass
class GridSConfig:
    # Number of tokens sampled per image (K).
    num_tokens: int = 16
    # Add the coordinate MLP embedding to the sampled tokens ("geometry injection").
    embed_coords: bool = True
    # MRoPE (t, h, w) of the sampled tokens: "continuous" puts h / w at the predicted
    # sub-token coordinate, "grid" lays the K tokens on a sqrt(K) x sqrt(K) grid, "first"
    # reuses the positions of the image's first K tokens.
    position_mode: str = "continuous"
    # Sample the DeepStack feature maps at the same coordinates (otherwise sampled tokens
    # get no DeepStack injection).
    sample_deepstack: bool = True
    # One sampler shared by all views, as in every Grid-Sampler implementation; False uses one
    # sampler per video key.
    share_across_views: bool = True
    # LayerNorm on the pooled feature before the coordinate MLP (``core_idea/grid_new.py``).
    input_layernorm: bool = False
    # Hidden sizes of the coordinate MLP and the coordinate embedding MLP.
    scout_hidden: int = 512
    coord_hidden: int = 256
    # Number of views, required when share_across_views is False.
    num_views: int = 1

    def __post_init__(self):
        if self.num_tokens < 1:
            raise ValueError("num_tokens must be >= 1")
        if self.position_mode not in POSITION_MODES:
            raise ValueError(f"position_mode must be one of {POSITION_MODES}")
        if self.position_mode == "grid" and round(self.num_tokens**0.5) ** 2 != self.num_tokens:
            raise ValueError("position_mode='grid' needs a square num_tokens")
        if self.num_views < 1:
            raise ValueError("num_views must be >= 1")

    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / CONFIG_FILENAME
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "GridSConfig":
        return cls(**json.loads((Path(directory) / CONFIG_FILENAME).read_text()))
