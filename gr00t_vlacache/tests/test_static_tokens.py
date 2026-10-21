# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for VLA-Cache token selection and the per-layer reuse schedule."""

import pytest
import torch

from gr00t_vlacache.config import VLACacheConfig
from gr00t_vlacache.static_tokens import (
    attention_entropy,
    layer_reuse_schedule,
    relevant_tokens,
    reusable_tokens,
    static_tokens,
    token_pixels,
    tokens_to_reuse,
)


def test_token_pixels_groups_merged_patches():
    # 2 images of 4 merged tokens, merge=2, 3 channels, temporal 2, patch 2 -> 24 features/patch.
    pv = torch.zeros(32, 3 * 2 * 2 * 2)
    pv[4:8] = 1.0  # all 4 patches of token 1 of image 0 are white
    tokens = token_pixels(pv, [4, 4], 2, (0.5,) * 3, (0.5,) * 3)
    assert [t.shape for t in tokens] == [torch.Size([4, 96])] * 2
    assert torch.allclose(tokens[0][1], torch.full((96,), 255.0))
    assert torch.allclose(tokens[0][0], torch.full((96,), 127.5))


def test_static_tokens_threshold_and_ratio():
    prev = torch.rand(8, 16) + 0.1
    cur = prev.clone()
    cur[0] = torch.rand(16)  # changed
    cur[3] = -prev[3]  # opposite
    static = static_tokens(cur, prev, threshold=0.996, ratio=0.5)
    assert len(static) == 4 and 0 not in static.tolist() and 3 not in static.tolist()
    assert len(static_tokens(cur, prev, 0.996, 1.0)) == 6


def test_static_tokens_zero_patch_not_static():
    prev = torch.zeros(2, 4)
    prev[1] = 1.0
    assert static_tokens(prev.clone(), prev, 0.996, 1.0).tolist() == [1]


def test_reusable_excludes_relevant_and_sorts():
    scores = torch.tensor([0.0, 9.0, 1.0, 8.0, 2.0, 3.0, 4.0, 5.0])
    relevant = relevant_tokens(scores, 0.25)
    assert sorted(relevant.tolist()) == [1, 3]
    assert reusable_tokens(torch.tensor([3, 0, 5, 1]), relevant).tolist() == [0, 5]


def test_schedule_matches_reference():
    # Reference get_layer_mask_schedule: reuse = 1 - minmax(entropy) = [2/3, 0, 1/3, 1], then
    # each upward step from the already smoothed previous value is scaled by 0.55.
    ent = torch.tensor([1.0, 3.0, 2.0, 0.0])
    reuse = layer_reuse_schedule(ent, 0.55)
    third = 0.55 / 3
    expected = [2 / 3, 0.0, third, third + (1.0 - third) * 0.55]
    assert torch.allclose(reuse, torch.tensor(expected), atol=1e-6)


def test_tokens_to_reuse_prefix():
    reusable = torch.arange(10, 20)
    assert tokens_to_reuse(reusable, 0.35).tolist() == [10, 11, 12]
    assert tokens_to_reuse(reusable, 0.0).tolist() == [10]
    assert tokens_to_reuse(reusable[:0], 0.5).numel() == 0


def test_attention_entropy():
    uniform = torch.full((2, 3, 4), 0.25)
    assert torch.isclose(attention_entropy(uniform), torch.log(torch.tensor(4.0)), atol=1e-5)
    onehot = torch.zeros(1, 2, 4)
    onehot[..., 0] = 1
    assert attention_entropy(onehot) < 1e-6


def test_config_validation():
    with pytest.raises(ValueError):
        VLACacheConfig(reuse_layers=(0, 2))
    with pytest.raises(ValueError):
        VLACacheConfig(static_ratio=1.5)
    assert VLACacheConfig(reuse_layers=[9, 2, 2]).reuse_layers == (2, 9)
