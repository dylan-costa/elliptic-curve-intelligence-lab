import numpy as np
import pytest

import eclab
from eclab.tasks import task1_label, task2_label, task2_mask


def test_package_imports():
    assert eclab.__version__


def test_task1_is_positive_rank():
    assert task1_label([0, 1, 2, 3, 0]).tolist() == [0, 1, 1, 1, 0]


def test_task2_keeps_only_ranks_0_and_2():
    ranks = np.array([0, 1, 2, 3, 4, 0, 2])
    assert task2_mask(ranks).tolist() == [True, False, True, False, False, True, True]


def test_task2_label_on_filtered_curves():
    ranks = np.array([0, 1, 2, 3, 4, 0, 2])
    kept = ranks[task2_mask(ranks)]
    assert task2_label(kept).tolist() == [0, 1, 0, 1]


def test_task2_label_rejects_odd_and_high_ranks():
    with pytest.raises(ValueError):
        task2_label([0, 1, 2])
    with pytest.raises(ValueError):
        task2_label([4])
