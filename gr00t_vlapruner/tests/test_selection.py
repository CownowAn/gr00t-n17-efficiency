# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for VLA-Pruner selection, checked against the reference implementation."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from gr00t_vlapruner.action_attention import image_cross_attention_blocks
from gr00t_vlapruner.config import VLAPrunerConfig
from gr00t_vlapruner.selection import combine_then_filter, redundancy_minimization, smoothed_history


def reference_redundancy_minimization(visual_feature_vectors, num_keep):
    """Verbatim logic of ``LlamaModel._redundancy_minization`` (VLA-Pruner, OpenVLA-OFT)."""
    if len(visual_feature_vectors) <= num_keep:
        return torch.arange(len(visual_feature_vectors))
    norm_matrix = visual_feature_vectors / visual_feature_vectors.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-6)
    cosine_matrix = 1.0 - torch.mm(norm_matrix, norm_matrix.t())
    selected = torch.empty(num_keep, dtype=torch.long)
    for i in range(num_keep):
        if i == 0:
            distances = cosine_matrix
            scores = torch.topk(distances, 2, dim=0, largest=False).values[1, :]
        else:
            distances = torch.index_select(cosine_matrix, 0, selected[:i])
            scores = torch.min(distances, dim=0).values
        selected[i] = torch.argmax(scores)
    return selected


def reference_combine(prefill_scores, action_scores, visual_features, num_keep):
    """Verbatim logic of ``_vlapruner_image_indices`` for the semantic_action mode."""
    prefill_topk = prefill_scores.topk(num_keep).indices
    action_topk = action_scores.topk(num_keep).indices
    candidate_indices = torch.unique(torch.cat([prefill_topk, action_topk])).sort().values
    if candidate_indices.numel() > num_keep:
        selected = visual_features.index_select(0, candidate_indices)
        final = reference_redundancy_minimization(selected, num_keep)
        candidate_indices = candidate_indices.index_select(0, final).sort().values
    return candidate_indices


@pytest.mark.parametrize("seed", range(5))
def test_matches_reference(seed):
    torch.manual_seed(seed)
    n, k = 64, 16
    semantic, action, feats = torch.rand(n), torch.rand(n), torch.randn(n, 32, dtype=torch.float64)
    ours = combine_then_filter(semantic, action, feats, k)["kept"]
    assert torch.equal(ours, reference_combine(semantic, action, feats, k))
    assert ours.numel() == k


def test_redundancy_matches_reference():
    torch.manual_seed(0)
    feats = torch.randn(40, 8, dtype=torch.float64)
    assert torch.equal(
        redundancy_minimization(feats, 10), reference_redundancy_minimization(feats, 10)
    )


def test_duplicates_are_not_picked_twice():
    feats = torch.tensor([[1.0, 0], [1.0, 0], [1.0, 0], [0, 1.0]])
    picked = redundancy_minimization(feats, 3)
    assert len(set(picked.tolist())) == 3


def test_union_within_budget_is_kept_whole():
    semantic = torch.tensor([9.0, 8, 0, 0, 0, 0])
    action = torch.tensor([9.0, 8, 0, 0, 0, 0])
    assert combine_then_filter(semantic, action, torch.randn(6, 3), 2)["kept"].tolist() == [0, 1]


def test_no_action_is_topk():
    semantic = torch.tensor([0.1, 0.9, 0.5, 0.7])
    assert combine_then_filter(semantic, None, torch.randn(4, 2), 2)["kept"].tolist() == [1, 3]


def test_smoothed_history():
    hist = deque(maxlen=3)
    hist.append(torch.tensor([1.0, 0.0]))
    hist.append(torch.tensor([0.0, 1.0]))
    assert smoothed_history(hist, 0.8) is None  # window not full yet -> no pruning
    hist.append(torch.tensor([2.0, 2.0]))
    # newest weight 1, then 0.8, then 0.64 (reference: weights[i] * av_hist[-1 - i])
    expected = (
        torch.tensor([2.0, 2.0]) + 0.8 * torch.tensor([0.0, 1.0]) + 0.64 * torch.tensor([1.0, 0.0])
    ) / 2.44
    torch.testing.assert_close(smoothed_history(hist, 0.8), expected)


def test_image_cross_attention_blocks():
    dit = SimpleNamespace(attend_text_every_n_blocks=2, transformer_blocks=list(range(32)))
    blocks = image_cross_attention_blocks(dit)
    assert blocks == [2, 6, 10, 14, 18, 22, 26, 30]
    assert [b for b in blocks if b >= 16] == [18, 22, 26, 30]


def test_config_validation():
    with pytest.raises(ValueError):
        VLAPrunerConfig(keep_ratio=0)
    with pytest.raises(ValueError):
        VLAPrunerConfig(action_blocks="first")
