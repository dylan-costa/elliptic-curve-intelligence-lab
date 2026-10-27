"""The two prediction tasks, defined once so that the code and the notes agree.

Task 1 asks whether a curve has positive rank. It is the obvious question, but the root
number w = (-1)^rank is a cheap function of the curve's bad reduction, so a model can
score well on it by learning parity alone.

Task 2 removes that shortcut: among curves of even rank (w = +1), is the rank 0 or 2?
Ranks 4 and up are left out, since they are rare and their analytic rank is not proven.
"""

import numpy as np

TASK2_RANKS = (0, 2)


def task1_label(rank) -> np.ndarray:
    """1 if the rank is positive, else 0."""
    return (np.asarray(rank) > 0).astype(np.int8)


def task2_mask(rank) -> np.ndarray:
    """Which curves belong to Task 2: those of rank 0 or 2."""
    return np.isin(np.asarray(rank), TASK2_RANKS)


def task2_label(rank) -> np.ndarray:
    """1 if the rank is 2, 0 if it is 0. Only defined on the curves in task2_mask."""
    rank = np.asarray(rank)
    if not task2_mask(rank).all():
        raise ValueError("task2_label is only defined for ranks 0 and 2; filter with task2_mask first")
    return (rank == 2).astype(np.int8)
