"""Features of a curve that a model is allowed to see, and the audit that says why.

    python -m eclab.features      # data/curves.parquet -> data/features.parquet

Two groups:

* ``invariants``: conductor, discriminant, j-invariant, torsion and reduction types.
* ``ap``: the Frobenius traces a_p = p + 1 - #E(F_p) for every prime p < AP_BOUND, by point
  counting. At bad primes the same count gives a_p = 1 (split multiplicative), -1 (non-split)
  or 0 (additive).

Everything excluded, and why, is in EXCLUDED; ``feature_columns`` can only return audited names.
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eclab.curves import b_invariants, discriminant, factor
from eclab.data.ecdata import DEFAULT_TABLE, load_curves

AP_BOUND = 100
PRIMES = [p for p in range(2, AP_BOUND) if all(p % q for q in range(2, int(p**0.5) + 1))]
AP_COLUMNS = [f"ap_{p}" for p in PRIMES]
DEFAULT_FEATURES = Path("data/features.parquet")

# allowed features: name -> what it is
INVARIANTS = {
    "log_conductor": "log N",
    "n_bad_primes": "number of primes dividing N",
    "conductor_is_prime": "N is prime",
    "semistable": "N is squarefree (no additive reduction)",
    "v2_conductor": "exponent of 2 in N",
    "v3_conductor": "exponent of 3 in N",
    "log_abs_disc": "log |minimal discriminant|",
    "disc_negative": "minimal discriminant < 0 (E(R) has one component)",
    "szpiro_ratio": "log|disc| / log N",
    "log_j": "log(1 + |j|)",
    "torsion_order": "#E(Q)_tors",
    "two_torsion": "#E(Q)[2], i.e. 1 + number of rational roots of the 2-division cubic",
    "n_split": "primes of split multiplicative reduction",
    "n_nonsplit": "primes of non-split multiplicative reduction",
    "n_additive": "primes of additive reduction",
}
AP_FEATURES = {c: f"a_p for p = {c[3:]}" for c in AP_COLUMNS}

# never features: name -> why
EXCLUDED = {
    "rank": "the label",
    "sha": "analytic Sha: computed from the BSD formula, which needs the rank; >1 for 20% of rank 0, 0.1% of rank 2",
    "root_number": "equals (-1)^rank for our labels; used only in the Step 4 parity sanity check",
    "regulator": "needs generators of E(Q), i.e. the rank",
    "leading_coefficient": "L^(r)(E,1)/r!; L(E,1) != 0 iff rank 0",
    "label": "identifier; the class letters are ordered by the a_p, so they are a coded feature",
    "iso_class": "identifier (same reason as label)",
    "iso_index": "identifier",
}
GROUPS = {"invariants": list(INVARIANTS), "ap": AP_COLUMNS}


def feature_columns(*groups: str) -> list[str]:
    """The columns of the named groups, e.g. feature_columns("invariants", "ap")."""
    cols = [c for g in groups for c in GROUPS[g]]
    assert not set(cols) & set(EXCLUDED)
    return cols


def ap_brute_force(ainvs, p: int) -> int:
    """a_p by counting affine solutions directly. Slow; the reference for tests."""
    a1, a2, a3, a4, a6 = ainvs
    n = sum(1 for x in range(p) for y in range(p)
            if (y * y + a1 * x * y + a3 * y - (x**3 + a2 * x * x + a4 * x + a6)) % p == 0)
    return p - n


def _legendre_table(p: int) -> np.ndarray:
    """chi[v] = Legendre symbol (v/p) for v in 0..p-1."""
    chi = -np.ones(p, dtype=np.int8)
    chi[0] = 0
    chi[(np.arange(1, p) ** 2) % p] = 1
    return chi


def ap_vectorized(b2: np.ndarray, b4: np.ndarray, b6: np.ndarray, p: int, chunk: int = 200_000) -> np.ndarray:
    """a_p for odd p from b2, b4, b6 already reduced mod p.

    Completing the square, y^2 + (a1 x + a3) y = x^3 + ... becomes Y^2 = g(x) with
    g(x) = 4x^3 + b2 x^2 + 2 b4 x + b6, so each x contributes 1 + chi(g(x)) points and
    a_p = p - #affine points = -sum_x chi(g(x)).
    """
    chi = _legendre_table(p)
    x = np.arange(p, dtype=np.int64)
    x2, x3 = x * x % p, x * x * x % p
    out = np.empty(len(b2), dtype=np.int8)
    for s in range(0, len(b2), chunk):
        sl = slice(s, s + chunk)
        g = (4 * x3 + b2[sl, None] * x2 + (2 * b4[sl, None] % p) * x + b6[sl, None]) % p
        out[sl] = -chi[g].sum(axis=1, dtype=np.int64)
    return out


def _mod(ints: list[int], p: int) -> np.ndarray:
    return np.fromiter((v % p for v in ints), dtype=np.int64, count=len(ints))


def ap_table(curves: pd.DataFrame) -> pd.DataFrame:
    a = [curves[c].astype(int).tolist() for c in ("a1", "a2", "a3")]
    a += [[int(v) for v in curves[c]] for c in ("a4", "a6")]
    b2, b4, b6, _ = zip(*(b_invariants(*row) for row in zip(*a)))
    out = {}
    for p in PRIMES:
        if p == 2:
            r = [_mod(col, 2) for col in a]
            n = sum(((y * y + r[0] * x * y + r[2] * y - (x**3 + r[1] * x * x + r[3] * x + r[4])) % 2 == 0)
                    for x in (0, 1) for y in (0, 1))
            out["ap_2"] = (2 - n).astype(np.int8)
        else:
            out[f"ap_{p}"] = ap_vectorized(_mod(b2, p), _mod(b4, p), _mod(b6, p), p)
    return pd.DataFrame(out, index=curves.index)


def invariant_table(curves: pd.DataFrame, ap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    factors_of: dict[int, dict] = {}
    ap_bad = {p: ap[f"ap_{p}"].to_numpy() for p in PRIMES}
    for i, row in enumerate(curves.itertuples(index=False)):
        ainvs = (int(row.a1), int(row.a2), int(row.a3), int(row.a4), int(row.a6))
        n = row.conductor
        f = factors_of.get(n) or factors_of.setdefault(n, factor(n))
        b2, b4, b6, _ = b_invariants(*ainvs)
        c4 = b2 * b2 - 24 * b4
        c6 = -b2**3 + 36 * b2 * b4 - 216 * b6
        delta = discriminant(*ainvs)
        split = nonsplit = additive = 0
        for p, e in f.items():
            if e > 1:
                additive += 1
                continue
            # multiplicative: counted directly for small p; for odd p, a_p = (-c6 / p)
            s = ap_bad[p][i] if p < AP_BOUND else (1 if pow(-c6 % p, (p - 1) // 2, p) == 1 else -1)
            split += s == 1
            nonsplit += s == -1
        log_disc = math.log(abs(delta))
        log_j_abs = 3 * math.log(abs(c4)) - log_disc if c4 else -math.inf
        rows.append((
            math.log(n), len(f), len(f) == 1 and next(iter(f.values())) == 1, additive == 0,
            f.get(2, 0), f.get(3, 0), log_disc, delta < 0, log_disc / math.log(n),
            max(log_j_abs, 0) + math.log1p(math.exp(-abs(log_j_abs))) if c4 else 0.0,
            row.torsion_order, _two_torsion(row.torsion), split, nonsplit, additive,
        ))
    return pd.DataFrame(rows, columns=list(INVARIANTS), index=curves.index)


def _two_torsion(torsion: str) -> int:
    """#E(Q)[2] from the invariant factors: one factor of 2 for each even invariant."""
    return 2 ** sum(int(t) % 2 == 0 for t in torsion.strip("[]").split(",") if t)


def build_features(curves: pd.DataFrame) -> pd.DataFrame:
    ap = ap_table(curves)
    inv = invariant_table(curves, ap)
    out = pd.concat([curves[["label"]], inv, ap], axis=1)
    for c in ("conductor_is_prime", "semistable", "disc_negative"):
        out[c] = out[c].astype(bool)
    return out.astype({c: np.int8 for c in ("n_bad_primes", "v2_conductor", "v3_conductor", "two_torsion",
                                            "n_split", "n_nonsplit", "n_additive")} | {"torsion_order": np.int16})


def load_features(path=DEFAULT_FEATURES) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m eclab.features` first")
    return pd.read_parquet(path)


def main():
    t0 = time.perf_counter()
    curves = load_curves(DEFAULT_TABLE)
    feats = build_features(curves)
    feats.to_parquet(DEFAULT_FEATURES, index=False)
    print(f"wrote {DEFAULT_FEATURES}: {len(feats):,} curves x {feats.shape[1] - 1} features "
          f"in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
