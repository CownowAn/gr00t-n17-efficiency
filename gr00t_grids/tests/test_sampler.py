# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the Grid Sampler module, its config and the launcher argument parsing."""

import numpy as np
import pytest
import torch

from gr00t_grids.config import GridSConfig
from gr00t_grids.launch_finetune import parse_grids_args
from gr00t_grids.sampler import ActiveTokenSampler, grid_sample_bilinear


def openpi_grid_sample(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """NumPy transcription of openpi ``grid_sample_jax`` (JAX clamps out-of-range gathers)."""
    h, w, _ = image.shape
    x = coords[:, 0] * (w - 1)
    y = coords[:, 1] * (h - 1)
    x0 = np.clip(np.floor(x).astype(np.int64), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, h - 1)
    x1, y1 = x0 + 1, y0 + 1

    def at(yy, xx):
        return image[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    return (
        wa[:, None] * at(y0, x0)
        + wb[:, None] * at(y1, x0)
        + wc[:, None] * at(y0, x1)
        + wd[:, None] * at(y1, x1)
    )


def test_matches_openpi_including_edges():
    rng = np.random.default_rng(0)
    feat = rng.normal(size=(2, 8, 6, 5))
    coords = rng.uniform(size=(2, 20, 2))
    coords[:, :4] = [[0, 0], [1, 1], [1, 0], [0, 1]]
    ours = grid_sample_bilinear(torch.tensor(feat), torch.tensor(coords)).numpy()
    ref = np.stack([openpi_grid_sample(feat[b], coords[b]) for b in range(2)])
    np.testing.assert_allclose(ours, ref, atol=1e-12)
    np.testing.assert_allclose(ours[:, 1], feat[:, -1, -1])  # bottom-right corner is exact


def test_grid_points_return_tokens():
    feat = torch.randn(1, 8, 8, 3, dtype=torch.float64)
    r, c = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    coords = torch.stack([c.flatten().double() / 7, r.flatten().double() / 7], -1)[None]
    torch.testing.assert_close(grid_sample_bilinear(feat, coords), feat.view(1, 64, 3))


def test_gradients_reach_coordinates_and_embedding():
    torch.manual_seed(0)
    sampler = ActiveTokenSampler(16, num_tokens=4)
    tokens, coords = sampler(torch.randn(3, 8, 8, 16))
    assert tokens.shape == (3, 4, 16) and coords.shape == (3, 4, 2)
    assert bool(((coords >= 0) & (coords <= 1)).all())
    (tokens**2).sum().backward()
    for name, p in sampler.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_layernorm_option():
    assert ActiveTokenSampler(8, input_layernorm=True).input_norm is not None
    assert ActiveTokenSampler(8, embed_coords=False).coord_encoder is None


def test_config_roundtrip_and_validation(tmp_path):
    cfg = GridSConfig(num_tokens=9, position_mode="grid", share_across_views=False, num_views=3)
    cfg.save(tmp_path)
    assert GridSConfig.load(tmp_path) == cfg
    with pytest.raises(ValueError):
        GridSConfig(num_tokens=8, position_mode="grid")
    with pytest.raises(ValueError):
        GridSConfig(position_mode="nope")


def test_launcher_args_pass_through():
    cfg, rest = parse_grids_args(
        [
            "--dataset_path",
            "/d",
            "--grids-num-tokens",
            "4",
            "--grids-embed-coords",
            "false",
            "--grids-position-mode",
            "grid",
            "--max_steps",
            "10",
        ]
    )
    assert (cfg.num_tokens, cfg.embed_coords, cfg.position_mode) == (4, False, "grid")
    assert rest == ["--dataset_path", "/d", "--max_steps", "10"]
