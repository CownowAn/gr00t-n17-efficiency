# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training pipeline that builds the usual GR00T N1.7 model and attaches a fresh Grid Sampler."""

import logging

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model import MODEL_REGISTRY
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
from gr00t.utils.dist_utils import run_or_wait_on_rank0

from .backbone import enable_grids
from .config import GridSConfig


class Gr00tN1d7GridSPipeline(Gr00tN1d7Pipeline):
    grids_config: GridSConfig = GridSConfig()

    def _create_model(self):
        # Base weights are loaded (and checked) by the original pipeline first; the sampler is
        # new, randomly initialized and trainable, as in openpi's grid training.
        model = super()._create_model()
        enable_grids(model.backbone, self.grids_config)
        n_params = sum(p.numel() for p in model.backbone.grid_sampler.parameters())
        logging.info(f"GridS: {self.grids_config}, sampler parameters: {n_params:,}")
        logging.info(
            "Trainable parameters with GridS: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )
        with run_or_wait_on_rank0(label="grids_config.json write") as is_rank0:
            if is_rank0:
                # experiment_cfg/ is copied into every checkpoint by CheckpointFormatCallback.
                self.grids_config.save(self.save_cfg_dir)
        return model


def install(config: GridSConfig) -> None:
    """Make ``experiment.run`` build GR00T N1.7 models through the GridS pipeline."""
    Gr00tN1d7GridSPipeline.grids_config = config
    MODEL_REGISTRY[Gr00tN1d7Config] = Gr00tN1d7GridSPipeline
