# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLA-Cache token selection (arXiv 2502.02175), at the granularity of Qwen3-VL merged tokens.

Mirrors ``experiments/robot/vla_cache_utils.py`` of the reference OpenVLA-OFT code:
``find_static_patches`` (pixel cosine similarity between frames), ``task_relevant_selection``
(drop the most attended tokens from the static set) and ``get_layer_mask_schedule``
(per-layer reuse proportion from normalized attention entropy).
"""

import torch
import torch.nn.functional as F


def token_pixels(
    pixel_values: torch.Tensor,
    counts: list[int],
    merge: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
) -> list[torch.Tensor]:
    """Raw pixels (0-255) of every merged token, one (n_tokens, features) tensor per image.

    ``pixel_values`` rows are ViT patches flattened as (channel, temporal, patch, patch), and the
    processor orders them so that each run of ``merge**2`` rows forms one merged LLM token.
    """
    n_channels = len(image_mean)
    mean = pixel_values.new_tensor(image_mean, dtype=torch.float32)[:, None]
    std = pixel_values.new_tensor(image_std, dtype=torch.float32)[:, None]
    patches = pixel_values.float().view(pixel_values.shape[0], n_channels, -1)
    raw = (patches * std + mean) * 255.0
    tokens = raw.reshape(-1, merge * merge * raw.shape[1] * raw.shape[2])
    return list(torch.split(tokens, counts))


def static_tokens(
    current: torch.Tensor, previous: torch.Tensor, threshold: float, ratio: float
) -> torch.Tensor:
    """Tokens of one image whose pixels barely changed, most similar first (at most ratio * n)."""
    sim = F.cosine_similarity(current, previous, dim=-1, eps=1e-8)
    top_k = round(ratio * current.shape[0])
    order = sim.argsort(descending=True, stable=True)
    order = order[sim[order] >= threshold]
    return order[:top_k]


def relevant_tokens(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    """Top ratio * n tokens of one image by text -> image attention."""
    return scores.argsort(descending=True, stable=True)[: round(ratio * scores.numel())]


def reusable_tokens(static: torch.Tensor, relevant: torch.Tensor) -> torch.Tensor:
    """Static tokens that are not task relevant, in ascending token order."""
    keep = ~torch.isin(static, relevant)
    return static[keep].sort().values


def attention_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Mean row entropy of the head-averaged attention map; ``probs`` is (heads, rows, keys)."""
    attn = probs.float().mean(dim=0)
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)
    attn = torch.nan_to_num(attn, nan=0.0)
    return -(attn * torch.log(attn + 1e-10)).sum(dim=-1).mean()


def layer_reuse_schedule(entropies: torch.Tensor, growth_factor: float = 0.55) -> torch.Tensor:
    """Per-layer reuse proportion: 1 - min-max normalized entropy, upward steps damped."""
    span = entropies.max() - entropies.min()
    reuse = (1.0 - (entropies - entropies.min()) / (span + 1e-10)).tolist()
    for i in range(1, len(reuse)):
        delta = reuse[i] - reuse[i - 1]
        if delta > 0:
            reuse[i] = reuse[i - 1] + delta * growth_factor
    return torch.tensor(reuse, dtype=torch.float32)


def tokens_to_reuse(reusable: torch.Tensor, proportion: float) -> torch.Tensor:
    """The prefix of the reusable list used at one layer (``max(1, int(p * n))`` as in VLA-Cache)."""
    if reusable.numel() == 0:
        return reusable
    return reusable[: max(1, int(proportion * reusable.numel()))]
