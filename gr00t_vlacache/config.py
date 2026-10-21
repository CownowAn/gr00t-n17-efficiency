# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field


@dataclass
class VLACacheConfig:
    # Master switch; False runs the full forward every time and keeps no cache.
    enabled: bool = True
    # (1) Static tokens: cosine similarity of the raw pixels of a token's region between the
    # current frame and the frame of the previous inference (the one the cache was built from).
    sim_threshold: float = 0.996
    # Share of each view's tokens that may be static (VLA-Cache: top_k=150 of 256 patches).
    static_ratio: float = 150 / 256
    # (2) Task-relevant tokens excluded from reuse: top share by the previous inference's
    # text -> image attention at ``relevance_layer`` (VLA-Cache: 100 of 256 at layer 15 of 32).
    relevant_ratio: float = 100 / 256
    relevance_layer: int = 7
    # (3) Layers at whose entry reusable tokens are removed from the queries (VLA-Cache
    # ``pruning_loc``); the removed set only grows.
    reuse_layers: tuple[int, ...] = (2, 6, 9, 11)
    # Smoothing of upward steps in the entropy-based per-layer reuse proportion.
    growth_factor: float = 0.55
    # Video modality keys in processor order; set from the policy's modality config.
    view_keys: list[str] = field(default_factory=list)
    # Per-channel normalization of pixel_values, to recover raw pixels; set from the processor.
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5)
    # Keep the per-inference selections on the backbone for inspection.
    record: bool = False

    def __post_init__(self):
        self.reuse_layers = tuple(sorted(set(int(i) for i in self.reuse_layers)))
        if not -1.0 <= self.sim_threshold <= 1.0:
            raise ValueError("sim_threshold must be in [-1, 1]")
        for name in ("static_ratio", "relevant_ratio", "growth_factor"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if any(i < 1 for i in self.reuse_layers):
            raise ValueError("reuse_layers must be >= 1 (layer 0 always runs on all tokens)")
