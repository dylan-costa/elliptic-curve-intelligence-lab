"""Train/test splits. Every split keeps each isogeny class on one side.

Isogenous curves share the conductor, the rank and every a_p, so a curve-level split would
test the model on near-copies of its training data. Assignment to folds hashes the class
name, so it does not depend on row order and is the same on every machine.

The named splits (each returns train and test masks over a frame with the columns
iso_class, conductor and conductor_is_prime):

* ``random``: conductor <= 500,000 (the complete range); test is 1 fold in 5.
* ``conductor``: complete range; train N <= 400,000, test 400,000 < N <= 500,000.
* ``prime``: prime conductors only; train N <= 10^8, test N > 2 * 10^8.
* ``family``: train on the complete range, test on prime N > 500,000.
"""

import numpy as np
import pandas as pd

COMPLETE_UP_TO = 500_000
HASH_KEY = "eclab-splits-v01"  # 16 bytes, as pandas requires; changing it reshuffles every split
N_FOLDS = 5
VALIDATION_FOLD = 1  # held out of training for early stopping and tuning; never fold 0, the random test fold


def class_fold(iso_class: pd.Series, k: int = N_FOLDS) -> np.ndarray:
    """A fold in 0..k-1 for each row, the same for every curve in an isogeny class."""
    h = pd.util.hash_array(iso_class.to_numpy(dtype=object), hash_key=HASH_KEY, categorize=True)
    return (h % np.uint64(k)).astype(np.int8)


def validation_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows of a training frame held out for early stopping: whole classes, by the same hash."""
    return class_fold(df.iso_class) == VALIDATION_FOLD


def split(df: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    n = df.conductor.to_numpy()
    complete = n <= COMPLETE_UP_TO
    prime = df.conductor_is_prime.to_numpy()
    if name == "random":
        test = class_fold(df.iso_class) == 0
        return complete & ~test, complete & test
    if name == "conductor":
        return n <= 400_000, complete & (n > 400_000)
    if name == "prime":
        return prime & (n <= 10**8), prime & (n > 2 * 10**8)
    if name == "family":
        return complete, prime & ~complete
    raise ValueError(f"unknown split {name!r}; choose from {SPLITS}")


SPLITS = ("random", "conductor", "prime", "family")
