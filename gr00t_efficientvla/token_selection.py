# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EfficientVLA visual token selection: task relevance (Sec. 3.3.1) and key / task / diversity
selection (Sec. 3.3.2-3.3.3), adapted to Qwen3-VL attention."""

import math

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half


def text_to_image_attention(
    layer: torch.nn.Module,
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    valid: torch.Tensor,
    text_pos: torch.Tensor,
    image_pos: torch.Tensor,
) -> torch.Tensor:
    """Raw relevance r_i = sum_j mean_h A[h, j, i] of ``layer`` for one sample (Eq. 2).

    The attention rows of the text queries are recomputed exactly as the layer sees them
    (input_layernorm, QK-norm, MRoPE, causal + padding mask, softmax over all keys), since
    FlashAttention does not return weights. Image tokens precede the task text, so the
    text -> image direction is the one a causal LLM actually has.

    Args:
        layer: a Qwen3-VL text decoder layer.
        hidden: (L, D) input hidden states of the layer.
        cos, sin: (L, head_dim) rotary embeddings of the sample.
        valid: (L,) bool, non-padding positions.
        text_pos: (T,) query positions.
        image_pos: (N,) image token positions.

    Returns:
        (N,) float32 relevance per image token.
    """
    attn = layer.self_attn
    head_dim = attn.head_dim
    x = layer.input_layernorm(hidden)
    q = attn.q_norm(attn.q_proj(x[text_pos]).view(text_pos.numel(), -1, head_dim))  # (T, Hq, d)
    k = attn.k_norm(attn.k_proj(x).view(x.shape[0], -1, head_dim))  # (L, Hkv, d)
    q = q * cos[text_pos, None] + rotate_half(q) * sin[text_pos, None]
    k = k * cos[:, None] + rotate_half(k) * sin[:, None]
    k = k.repeat_interleave(attn.num_key_value_groups, dim=1)  # (L, Hq, d)

    logits = torch.einsum("thd,lhd->htl", q.float(), k.float()) * attn.scaling  # (Hq, T, L)
    pos = torch.arange(x.shape[0], device=x.device)
    allowed = valid[None, :] & (pos[None, :] <= text_pos[:, None])  # (T, L)
    probs = logits.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
    return probs[:, :, image_pos].mean(dim=0).sum(dim=0)


def min_max(x: torch.Tensor) -> torch.Tensor:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else torch.zeros_like(x)


def select_key_task_diverse(
    scores: torch.Tensor,
    features: torch.Tensor,
    k_final: int,
    num_key_tokens: int = 4,
    alpha: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Indices (into one view) of V_key, V_task and V_div, plus their sorted union "kept".

    V_key: top ``num_key_tokens`` by relevance. V_task: next floor(alpha * K_aug) by relevance.
    V_div: the K_div remaining tokens with the largest cosine distance to V_key (Eq. 4).
    """
    n = scores.numel()
    k_final = min(k_final, n)
    k_key = min(num_key_tokens, k_final)
    k_aug = k_final - k_key
    k_task = math.floor(alpha * k_aug)
    k_div = k_aug - k_task
    s = min_max(scores.float())

    order = s.argsort(descending=True)
    key = order[:k_key]
    task = order[k_key : k_key + k_task]
    taken = torch.zeros(n, dtype=torch.bool, device=scores.device)
    taken[key] = True
    taken[task] = True

    if k_div > 0:
        if k_key > 0:
            f = F.normalize(features.float(), dim=-1)
            diversity = 1.0 - (f @ f[key].T).max(dim=-1).values
        else:  # no anchor to be diverse from: fall back to relevance
            diversity = s.clone()
        div = diversity.masked_fill(taken, float("-inf")).topk(k_div).indices
    else:
        div = key.new_empty(0)
    kept = torch.cat([key, task, div]).sort().values
    return {"key": key, "task": task, "div": div, "kept": kept, "scores": s}
