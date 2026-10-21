# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differentiable Grid Sampler (GridS, arXiv 2605.11817), ported from openpi ``models/grid.py``."""

import torch
from torch import nn


def grid_sample_bilinear(feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Bilinear sampling at normalized [0, 1] coordinates, matching openpi ``grid_sample_jax``.

    ``x = coords[..., 0] * (W - 1)``, ``y = coords[..., 1] * (H - 1)``. As in the JAX version,
    the weights use the unclamped ``x0 + 1`` / ``y0 + 1`` while the gathered indices are
    clamped, so coordinates on the right / bottom edge return that edge exactly (the PyTorch
    port in ``core_idea/grid_pytorch.py`` clamps both and returns 0 there).

    Args:
        feat: (B, H, W, C) feature maps.
        coords: (B, K, 2) normalized (x, y) coordinates.

    Returns:
        (B, K, C) sampled features.
    """
    batch, height, width, _ = feat.shape
    x = coords[..., 0] * (width - 1)
    y = coords[..., 1] * (height - 1)
    x0 = x.floor().clamp(0, width - 1)
    y0 = y.floor().clamp(0, height - 1)
    x1, y1 = x0 + 1, y0 + 1

    ix0, iy0 = x0.long(), y0.long()
    ix1, iy1 = ix0.add(1).clamp(max=width - 1), iy0.add(1).clamp(max=height - 1)
    b = torch.arange(batch, device=feat.device)[:, None].expand_as(ix0)
    ia, ib = feat[b, iy0, ix0], feat[b, iy1, ix0]
    ic, id_ = feat[b, iy0, ix1], feat[b, iy1, ix1]

    wa = ((x1 - x) * (y1 - y))[..., None]
    wb = ((x1 - x) * (y - y0))[..., None]
    wc = ((x - x0) * (y1 - y))[..., None]
    wd = ((x - x0) * (y - y0))[..., None]
    return wa * ia + wb * ib + wc * ic + wd * id_


class ActiveTokenSampler(nn.Module):
    """Pooled feature -> K normalized coordinates -> bilinear samples (+ coordinate embedding)."""

    def __init__(
        self,
        vision_dim: int,
        num_tokens: int = 16,
        embed_coords: bool = True,
        input_layernorm: bool = False,
        scout_hidden: int = 512,
        coord_hidden: int = 256,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.input_norm = nn.LayerNorm(vision_dim) if input_layernorm else None
        self.scout_mlp = nn.Sequential(
            nn.Linear(vision_dim, scout_hidden),
            nn.ReLU(),
            nn.Linear(scout_hidden, num_tokens * 2),
            nn.Sigmoid(),
        )
        self.coord_encoder = (
            nn.Sequential(
                nn.Linear(2, coord_hidden), nn.ReLU(), nn.Linear(coord_hidden, vision_dim)
            )
            if embed_coords
            else None
        )

    def predict_coords(self, feature_map: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) -> (B, K, 2) coordinates in [0, 1]."""
        pooled = feature_map.mean(dim=(1, 2))
        if self.input_norm is not None:
            pooled = self.input_norm(pooled)
        return self.scout_mlp(pooled).view(feature_map.shape[0], self.num_tokens, 2)

    def forward(self, feature_map: torch.Tensor, coords: torch.Tensor | None = None):
        """Returns sampled tokens (B, K, C) and the coordinates used (B, K, 2)."""
        param = next(self.parameters())
        feature_map = feature_map.to(param.dtype)
        if coords is None:
            coords = self.predict_coords(feature_map)
        sampled = grid_sample_bilinear(feature_map, coords)
        if self.coord_encoder is not None:
            sampled = sampled + self.coord_encoder(coords.to(param.dtype))
        return sampled, coords
