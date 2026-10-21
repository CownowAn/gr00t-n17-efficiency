# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for budget allocation and top-k selection (no checkpoint needed)."""

import pytest
import torch

from gr00t_adp.config import ADPConfig
from gr00t_adp.token_scoring import allocate_budget, select_tokens


COUNTS = [64, 64, 64, 64, 64]


@pytest.mark.parametrize("ratio", [0.1, 0.3, 0.5, 0.75, 1.0])
def test_uniform_budget(ratio):
    budget = allocate_budget(COUNTS, ratio, "per_view_uniform")
    assert budget == [max(1, round(64 * ratio))] * 5


@pytest.mark.parametrize("ratio", [0.01, 0.3, 0.5, 0.9])
def test_weighted_budget_sums_and_bounds(ratio):
    weights = [5.0, 0.0, 1.0, 1.0, 100.0]
    budget = allocate_budget(COUNTS, ratio, "per_view_weighted", weights, min_per_view=2)
    assert sum(budget) == max(10, round(320 * ratio))
    assert all(2 <= k <= 64 for k in budget)


def test_weighted_budget_redistributes_overflow():
    # The heavy view saturates at 64; the overflow follows the remaining weights (2:1:1),
    # and zero-weight views stay at the minimum.
    budget = allocate_budget(COUNTS, 0.5, "per_view_weighted", [100, 2, 1, 1, 0], min_per_view=1)
    assert budget == [64, 47, 24, 24, 1]


def test_weighted_budget_all_saturated():
    budget = allocate_budget(COUNTS, 1.0, "per_view_weighted", [1, 2, 3, 4, 5])
    assert budget == COUNTS


def test_weighted_matches_adp_split():
    # ADP's LIBERO default: 2 x 256 tokens, keep 0.75, split [0.4, 0.6] -> 154 / 230.
    assert allocate_budget([256, 256], 0.75, "per_view_weighted", [0.4, 0.6]) == [154, 230]


def test_select_topk_and_sorted():
    scores = [torch.tensor([0.1, 0.9, 0.5, 0.7]), torch.tensor([3.0, 2.0, 1.0, 0.0])]
    kept = select_tokens(scores, 0.5, "per_view_uniform")
    assert [k.tolist() for k in kept] == [[1, 3], [0, 1]]


def test_global_keeps_min_per_view():
    scores = [torch.full((8,), -10.0), torch.arange(8.0)]
    kept = select_tokens(scores, 0.5, "global", min_per_view=2)
    assert len(kept[0]) == 2 and len(kept[1]) == 6
    assert kept[1].tolist() == [2, 3, 4, 5, 6, 7]


def test_config_validation():
    with pytest.raises(ValueError):
        ADPConfig(keep_ratio=0.0)
    with pytest.raises(ValueError):
        ADPConfig(budget_policy="nope")
