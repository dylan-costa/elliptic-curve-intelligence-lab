"""The table every model reads: the features plus the columns that the tasks and splits need."""

import numpy as np
import pandas as pd

from eclab.data.ecdata import load_curves
from eclab.features import AP_COLUMNS, load_features, mestre_nagao_sum
from eclab.splits import split
from eclab.tasks import task1_label, task2_label, task2_mask

TASKS = (1, 2)


def load_frame() -> pd.DataFrame:
    """One row per curve: iso_class, conductor, rank, every feature, and the derived mn_sum."""
    curves = load_curves()
    feats = load_features()
    if not (feats.label.to_numpy() == curves.label.to_numpy()).all():
        raise ValueError("curves and features are not in the same order; rebuild data/features.parquet")
    frame = pd.concat([curves[["iso_class", "conductor", "rank"]], feats.drop(columns="label")], axis=1)
    frame["mn_sum"] = mestre_nagao_sum(frame[AP_COLUMNS].to_numpy(), frame.conductor.to_numpy())
    return frame


def task_split(frame: pd.DataFrame, task: int, name: str):
    """(train rows, test rows, train labels, test labels) for one task and one named split."""
    train, test = split(frame, name)
    if task == 2:
        keep = task2_mask(frame["rank"].to_numpy())
        train, test = train & keep, test & keep
    elif task != 1:
        raise ValueError(f"unknown task {task!r}; choose from {TASKS}")
    label = task1_label if task == 1 else task2_label
    tr, te = frame[train], frame[test]
    return tr, te, label(tr["rank"].to_numpy()), label(te["rank"].to_numpy())


def class_codes(iso_class: pd.Series) -> np.ndarray:
    """Integer code per row, equal within an isogeny class (the unit the bootstrap resamples)."""
    return pd.factorize(iso_class)[0]
