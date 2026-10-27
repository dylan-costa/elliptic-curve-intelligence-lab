"""Read the LMFDB curve export into a table.

Each line of the source file is

    label|[a1,a2,a3,a4,a6]|rank|[torsion invariants]|conductor|sha

for example ``11.a1|[0,-1,1,-7820,-263580]|0|[]|11|1``. The label is
``<conductor>.<isogeny class letters><index in class>``.

a1, a2, a3 of a reduced minimal model lie in {-1, 0, 1}, so they fit in int8. a4 and a6 can
exceed 64 bits (the largest has 85), so they are kept as decimal strings; use
``ainvs(row)`` to get exact Python integers.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SOURCE = Path("data/raw/ecdata_v2.txt")
DEFAULT_TABLE = Path("data/curves.parquet")


def source_path() -> Path:
    """The source file: $ECLAB_ECDATA if set, else data/raw/ecdata_v2.txt."""
    return Path(os.environ.get("ECLAB_ECDATA", DEFAULT_SOURCE))


def parse_line(line: str) -> tuple:
    label, ainvs, rank, torsion, conductor, sha = line.rstrip("\n").split("|")
    a1, a2, a3, a4, a6 = ainvs.strip("[]{}").split(",")
    inv = [int(t) for t in torsion.strip("[]{}").split(",") if t]
    order = 1
    for t in inv:
        order *= t
    cond, rest = label.split(".")
    letters = rest.rstrip("0123456789")
    return (
        label, f"{cond}.{letters}", int(rest[len(letters):]), int(conductor),
        int(a1), int(a2), int(a3), a4, a6,
        int(rank), str(inv).replace(" ", ""), order, int(sha),
    )


COLUMNS = ["label", "iso_class", "iso_index", "conductor",
           "a1", "a2", "a3", "a4", "a6", "rank", "torsion", "torsion_order", "sha"]

DTYPES = {"iso_index": np.int16, "conductor": np.int64, "a1": np.int8, "a2": np.int8, "a3": np.int8,
          "rank": np.int8, "torsion_order": np.int16, "sha": np.int64}


def read_ecdata(path) -> pd.DataFrame:
    """Parse the source file. The a1, a2, a3 range is checked here, since the int8 cast would hide it."""
    with open(path, encoding="utf-8") as f:
        rows = [parse_line(line) for line in f if line.strip()]
    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in ("a1", "a2", "a3"):
        bad = ~df[col].between(-1, 1)
        if bad.any():
            raise ValueError(f"{bad.sum()} curves have {col} outside {{-1,0,1}}, e.g. {df.label[bad].iloc[0]}")
    return df.astype(DTYPES)


def ainvs(row) -> tuple[int, int, int, int, int]:
    """Exact a-invariants of one row (a pandas row or namedtuple)."""
    return int(row.a1), int(row.a2), int(row.a3), int(row.a4), int(row.a6)


def load_curves(path=DEFAULT_TABLE) -> pd.DataFrame:
    """The parsed table written by ``python -m eclab.data.build``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m eclab.data.build` first")
    return pd.read_parquet(path)
