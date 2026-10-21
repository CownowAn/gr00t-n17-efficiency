# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLA-Pruner token selection (arXiv 2511.16449), ported from the reference OpenVLA-OFT code
(``transformers/.../llama/modeling_llama.py``: ``_vlapruner_image_indices`` and
``_redundancy_minization``; ``modeling_prismatic.py``: ``_build_historical_action_attention``)."""

from collections import deque

import torch


def smoothed_history(history: deque, decay: float) -> torch.Tensor | None:
    """Decaying window average: the newest entry has weight 1, older ones decay ** age."""
    if len(history) < history.maxlen:
        return None
    weights = [decay**age for age in range(len(history))]
    guided = sum(w * h.float() for w, h in zip(weights, reversed(history)))
    return guided / sum(weights)


def redundancy_minimization(features: torch.Tensor, num_keep: int) -> torch.Tensor:
    """Greedy max-min cosine-distance selection of ``num_keep`` rows of ``features``.

    The first pick is the row farthest from its nearest neighbour; each next pick maximizes the
    distance to the closest already selected row. Importance scores are not used. Unlike the
    reference, already selected rows are excluded explicitly, so duplicate rows cannot be picked
    twice (the reference only avoids them through their zero self-distance).
    """
    n = features.shape[0]
    if n <= num_keep:
        return torch.arange(n, device=features.device)
    f = features.float()
    f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-6)
    distance = 1.0 - f @ f.T
    selected = torch.empty(num_keep, dtype=torch.long, device=features.device)
    taken = torch.zeros(n, dtype=torch.bool, device=features.device)
    for i in range(num_keep):
        if i == 0:
            scores = torch.topk(distance, 2, dim=0, largest=False).values[1]
        else:
            scores = distance[selected[:i]].min(dim=0).values
        scores = scores.masked_fill(taken, float("-inf"))
        selected[i] = scores.argmax()
        taken[selected[i]] = True
    return selected


def combine_then_filter(
    semantic: torch.Tensor,
    action: torch.Tensor | None,
    features: torch.Tensor,
    num_keep: int,
) -> dict[str, torch.Tensor]:
    """Kept token indices of one view (ascending), plus the intermediate sets.

    Candidates are the union of the top ``num_keep`` by semantic and by action relevance; if the
    union is larger than the budget, ``redundancy_minimization`` on the tokens' LLM input
    embeddings picks ``num_keep`` of them. Without action relevance it is plain top-k.
    """
    num_keep = min(num_keep, semantic.numel())
    semantic_top = semantic.topk(num_keep).indices
    if action is None:
        kept = semantic_top.sort().values
        return {"semantic_top": semantic_top, "kept": kept}
    action_top = action.topk(num_keep).indices
    candidates = torch.unique(torch.cat([semantic_top, action_top])).sort().values
    kept = candidates
    if candidates.numel() > num_keep:
        pick = redundancy_minimization(features[candidates], num_keep)
        kept = candidates[pick].sort().values
    return {
        "semantic_top": semantic_top,
        "action_top": action_top,
        "candidates": candidates,
        "kept": kept,
    }
