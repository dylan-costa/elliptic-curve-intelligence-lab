"""Consistency checks on the curve table. Each check counts the curves that fail it.

None of these prove the ranks are right; they catch parsing errors, truncated files and
rows that do not describe what the label says.
"""

from collections import Counter

import pandas as pd

from eclab.curves import discriminant, factor

# Mazur (1977): the 15 possible torsion groups of E(Q), as invariant factors
MAZUR_TORSION = {"[]", "[2]", "[3]", "[4]", "[5]", "[6]", "[7]", "[8]", "[9]", "[10]", "[12]",
                 "[2,2]", "[2,4]", "[2,6]", "[2,8]"}

# the largest possible exponent of p in a conductor: 8 for p = 2, 5 for p = 3, 2 otherwise
MAX_CONDUCTOR_EXPONENT = {2: 8, 3: 5}


def check_model(a1, a2, a3, a4, a6, conductor_factors) -> str | None:
    """None if the model is consistent with its conductor, else the name of the first failure.

    For a global minimal model, the primes of bad reduction are exactly the primes dividing
    the discriminant, so Δ with the conductor's primes divided out must be ±1.
    """
    delta = discriminant(a1, a2, a3, a4, a6)
    if delta == 0:
        return "singular"
    for p in conductor_factors:
        if delta % p:
            return "bad_prime_not_in_discriminant"
        while delta % p == 0:
            delta //= p
    if abs(delta) != 1:
        return "discriminant_has_good_prime"
    return None


def validate(df: pd.DataFrame) -> dict:
    """Run every check. Returns {check name: number of failing curves} plus a few examples."""
    fails: Counter = Counter()
    examples: dict[str, str] = {}

    def fail(name, mask):
        n = int(mask.sum())
        fails[name] += n
        if n and name not in examples:
            examples[name] = df.label[mask].iloc[0]

    fail("duplicate_label", df.label.duplicated())
    fail("label_conductor_mismatch", df.label.str.split(".").str[0].astype("int64") != df.conductor)
    fail("not_reduced_model", ~df.a1.isin([0, 1]) | ~df.a3.isin([0, 1]) | ~df.a2.isin([-1, 0, 1]))
    fail("torsion_not_in_mazur_list", ~df.torsion.isin(MAZUR_TORSION))
    root = df.sha.pow(0.5).round().astype("int64")
    fail("sha_not_a_positive_square", (df.sha < 1) | (root * root != df.sha))
    fail("rank_varies_in_isogeny_class", df.groupby("iso_class")["rank"].transform("nunique") > 1)

    factors_of = {}
    bad_exponent = []
    model_fail = []
    for row in df.itertuples(index=False):
        n = row.conductor
        if n not in factors_of:
            factors_of[n] = factor(n)
        f = factors_of[n]
        bad_exponent.append(any(e > MAX_CONDUCTOR_EXPONENT.get(p, 2) for p, e in f.items()))
        model_fail.append(check_model(row.a1, row.a2, row.a3, int(row.a4), int(row.a6), f))
    fail("conductor_exponent_too_large", pd.Series(bad_exponent, index=df.index))
    model_fail = pd.Series(model_fail, index=df.index)
    for name in ("singular", "bad_prime_not_in_discriminant", "discriminant_has_good_prime"):
        fail(name, model_fail == name)

    return {"curves": len(df), "failures": dict(fails), "examples": examples}
