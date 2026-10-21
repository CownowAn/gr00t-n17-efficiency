# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text-driven visual token scoring and per-view budget allocation (ADP, Qwen3-VL variant)."""

import math

import torch


def text_to_image_scores(
    layer: torch.nn.Module,
    hidden: torch.Tensor,
    text_pos: torch.Tensor,
    image_pos: torch.Tensor,
) -> torch.Tensor:
    """Mean pre-softmax attention logit from text queries to each image key in ``layer``.

    Mirrors ``_compute_qk_importance_generic`` in VLA-ADP, adapted to Qwen3-VL attention:
    input_layernorm -> q_proj/k_proj -> per-head q_norm/k_norm (QK-norm) -> GQA expansion.
    RoPE and softmax are not applied, as in the original.

    Args:
        layer: a Qwen3-VL text decoder layer (layer 0 for the input embeddings).
        hidden: (L, D) input hidden states of one sample.
        text_pos: (T,) positions of query text tokens.
        image_pos: (N,) positions of image tokens.

    Returns:
        (N,) float32 importance per image token.
    """
    attn = layer.self_attn
    head_dim = attn.head_dim
    h = layer.input_layernorm(hidden[torch.cat([text_pos, image_pos])])
    h_text, h_image = h[: text_pos.numel()], h[text_pos.numel() :]

    q = attn.q_norm(attn.q_proj(h_text).view(h_text.shape[0], -1, head_dim))  # (T, Hq, d)
    k = attn.k_norm(attn.k_proj(h_image).view(h_image.shape[0], -1, head_dim))  # (N, Hkv, d)
    # Query head h reads KV head h // num_key_value_groups, as in transformers' repeat_kv.
    k = k.repeat_interleave(attn.num_key_value_groups, dim=1)  # (N, Hq, d)

    scores = torch.einsum("thd,nhd->htn", q.float(), k.float()) * attn.scaling  # (Hq, T, N)
    return scores.mean(dim=(0, 1))


def allocate_budget(
    counts: list[int],
    keep_ratio: float,
    policy: str,
    weights: list[float] | None = None,
    min_per_view: int = 1,
) -> list[int] | None:
    """Number of tokens to keep per image. Returns ``None`` for the ``global`` policy."""
    mins = [min(min_per_view, n) for n in counts]
    if policy == "global":
        return None
    if policy == "per_view_uniform":
        return [max(m, min(n, round(n * keep_ratio))) for n, m in zip(counts, mins)]
    if policy != "per_view_weighted":
        raise ValueError(f"Unknown budget policy: {policy}")

    if weights is None or len(weights) != len(counts):
        weights = [1.0] * len(counts)
    weights = [max(0.0, float(w)) for w in weights]
    if sum(weights) <= 0:
        weights = [1.0] * len(counts)
    total = sum(counts)
    keep_total = max(sum(mins), min(total, round(total * keep_ratio)))

    # Water-filling: shares proportional to weights; a view that hits its bound (count or
    # min) is fixed there and the rest of the budget is re-split among the other views.
    raw = [0.0] * len(counts)
    fixed: set[int] = set()
    while len(fixed) < len(counts):
        free = [i for i in range(len(counts)) if i not in fixed]
        budget = keep_total - sum(raw[i] for i in fixed)
        wsum = sum(weights[i] for i in free)
        for i in free:
            raw[i] = budget * weights[i] / wsum if wsum > 0 else budget / len(free)
        over = [i for i in free if raw[i] > counts[i]]
        under = [i for i in free if raw[i] < mins[i]]
        for i in over:
            raw[i] = float(counts[i])
        for i in under if not over else []:
            raw[i] = float(mins[i])
        if not over and not under:
            break
        fixed.update(over or under)

    keep = [max(m, min(n, math.floor(r))) for r, n, m in zip(raw, counts, mins)]
    # Fix rounding drift: add where the ideal share is most underserved, remove where it is
    # most overserved, never leaving [min, count].
    while sum(keep) < keep_total:
        i = max(
            (i for i in range(len(keep)) if keep[i] < counts[i]),
            key=lambda i: raw[i] - keep[i],
        )
        keep[i] += 1
    while sum(keep) > keep_total:
        i = min(
            (i for i in range(len(keep)) if keep[i] > mins[i]),
            key=lambda i: raw[i] - keep[i],
        )
        keep[i] -= 1
    return keep


def select_tokens(
    scores: list[torch.Tensor],
    keep_ratio: float,
    policy: str,
    weights: list[float] | None = None,
    min_per_view: int = 1,
) -> list[torch.Tensor]:
    """Top-k selection per image. Returns sorted local indices kept in each image."""
    counts = [s.numel() for s in scores]
    budget = allocate_budget(counts, keep_ratio, policy, weights, min_per_view)
    if budget is not None:
        return [
            torch.topk(s, k=k, sorted=False).indices.sort().values for s, k in zip(scores, budget)
        ]

    # Global: guarantee min_per_view per image first, then fill by global rank.
    total = sum(counts)
    mins = [min(min_per_view, n) for n in counts]
    keep_total = max(sum(mins), min(total, round(total * keep_ratio)))
    offsets = [0]
    for n in counts:
        offsets.append(offsets[-1] + n)
    flat = torch.cat(scores)
    chosen = torch.zeros(total, dtype=torch.bool, device=flat.device)
    for i, m in enumerate(mins):
        if m > 0:
            chosen[offsets[i] + torch.topk(scores[i], k=m).indices] = True
    remaining = keep_total - int(chosen.sum())
    if remaining > 0:
        rest = flat.masked_fill(chosen, float("-inf"))
        chosen[torch.topk(rest, k=remaining).indices] = True
    return [chosen[offsets[i] : offsets[i + 1]].nonzero().squeeze(-1) for i in range(len(counts))]
