# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for EfficientVLA token selection, config and DiT step recovery."""

import json

import pytest
import torch

from gr00t_efficientvla.config import EfficientVLAConfig, drop_layers_from_importance
from gr00t_efficientvla.dit_cache import denoising_step
from gr00t_efficientvla.token_selection import select_key_task_diverse


def test_split_sizes():
    torch.manual_seed(0)
    sel = select_key_task_diverse(torch.rand(64), torch.randn(64, 8), 32, num_key_tokens=4)
    # K_aug = 28 -> K_task = 14, K_div = 14
    assert (len(sel["key"]), len(sel["task"]), len(sel["div"])) == (4, 14, 14)
    assert sel["kept"].tolist() == sorted(set(sel["kept"].tolist()))
    assert len(sel["kept"]) == 32


def test_key_and_task_are_top_scores():
    scores = torch.arange(10.0)
    sel = select_key_task_diverse(scores, torch.randn(10, 4), 6, num_key_tokens=2, alpha=0.5)
    assert sorted(sel["key"].tolist()) == [8, 9]
    assert sorted(sel["task"].tolist()) == [6, 7]


def test_alpha_one_is_plain_topk():
    torch.manual_seed(1)
    scores = torch.rand(50)
    sel = select_key_task_diverse(scores, torch.randn(50, 4), 20, num_key_tokens=4, alpha=1.0)
    assert sel["kept"].tolist() == scores.topk(20).indices.sort().values.tolist()


def test_diversity_picks_far_tokens():
    # Token 0 is the key; tokens 1-3 are copies of it, token 4 is orthogonal, token 5 opposite.
    feats = torch.tensor([[1.0, 0], [1, 0], [1, 0.01], [1, 0], [0, 1], [-1, 0]])
    scores = torch.tensor([9.0, 8, 7, 6, 0, 0])
    sel = select_key_task_diverse(scores, feats, 3, num_key_tokens=1, alpha=0.0)
    assert sel["key"].tolist() == [0]
    assert sorted(sel["div"].tolist()) == [4, 5]


def test_budget_smaller_than_key():
    sel = select_key_task_diverse(torch.arange(8.0), torch.randn(8, 2), 2, num_key_tokens=4)
    assert sel["kept"].tolist() == [6, 7]


@pytest.mark.parametrize("num_steps,num_buckets", [(4, 1000), (10, 1000), (3, 1000), (4, 10)])
def test_denoising_step_roundtrip(num_steps, num_buckets):
    for step in range(num_steps):
        t = torch.full((2,), int(step / num_steps * num_buckets))
        assert denoising_step(t, num_steps, num_buckets) == step


def test_config(tmp_path):
    assert EfficientVLAConfig(drop_layers=[5, 3, 5]).drop_layers == (3, 5)
    off = EfficientVLAConfig.disabled()
    assert (off.drop_layers, off.keep_ratio, off.cache_interval) == ((), 1.0, 1)
    with pytest.raises(ValueError):
        EfficientVLAConfig(cache_interval=0)
    path = tmp_path / "imp.json"
    path.write_text(json.dumps({"ranking": [9, 2, 14, 0]}))
    assert drop_layers_from_importance(str(path), 3) == (2, 9, 14)
