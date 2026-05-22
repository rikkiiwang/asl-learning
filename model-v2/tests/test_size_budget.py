import pytest
import torch.nn as nn
from aslv2.size_budget import BUDGETS, count_params, assert_within_budget, total_hard_cap_mb


def test_budgets_cover_all_four_stages():
    assert set(BUDGETS) == {"detector", "landmark", "recognizer", "appearance"}
    assert total_hard_cap_mb() == 20.0


def test_small_module_within_budget():
    tiny = nn.Linear(10, 10)                              # 110 params
    assert_within_budget("detector", tiny)               # no raise


def test_oversized_module_raises():
    big = nn.Linear(4000, 4000)                          # ~16M params > 2M cap
    with pytest.raises(AssertionError, match="detector"):
        assert_within_budget("detector", big)


def test_count_params_counts_trainable():
    assert count_params(nn.Linear(10, 10)) == 110
