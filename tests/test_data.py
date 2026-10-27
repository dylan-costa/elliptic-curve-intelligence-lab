from pathlib import Path

import pytest

from eclab.curves import discriminant, factor, is_probable_prime
from eclab.data.ecdata import DEFAULT_TABLE, ainvs, load_curves, parse_line, read_ecdata
from eclab.data.validate import validate

FIXTURE = Path(__file__).parent / "fixtures" / "ecdata_sample.txt"


@pytest.fixture
def sample():
    return read_ecdata(FIXTURE)


def test_parse_line():
    row = parse_line("15.a2|[1,1,1,-135,-660]|0|[2,2]|15|1\n")
    assert row == ("15.a2", "15.a", 2, 15, 1, 1, 1, "-135", "-660", 0, "[2,2]", 4, 1)


def test_parse_accepts_brace_format():
    # ecdata.txt (the older copy) uses {} instead of []
    assert parse_line("11.a2|{0,-1,1,-10,-20}|0|{5}|11|1") == parse_line("11.a2|[0,-1,1,-10,-20]|0|[5]|11|1")


def test_read_sample(sample):
    assert len(sample) == 21
    assert sample.label.is_unique
    assert set(sample.iso_class[sample.conductor == 14]) == {"14.a"}


def test_big_coefficients_stay_exact(sample):
    row = sample[sample.label == "21758.i1"].iloc[0]
    assert ainvs(row)[4] == -32992152521343165084  # does not fit in int64


@pytest.mark.parametrize("label, delta", [
    # LMFDB orders a class by its own rule, so 11.a1 is Cremona's 11a2, not X_0(11)
    ("11.a3", -11), ("11.a2", -11**5), ("11.a1", -11), ("37.a1", 37), ("389.a1", 389), ("5077.a1", 5077),
])
def test_discriminants_of_famous_curves(sample, label, delta):
    row = sample[sample.label == label].iloc[0]
    assert discriminant(*ainvs(row)) == delta


def test_factor():
    assert factor(1) == {}
    assert factor(508032) == {2: 7, 3: 4, 7: 2}
    assert factor(299996953) == {299996953: 1}
    assert factor(2 * 3 * 1000003**2) == {2: 1, 3: 1, 1000003: 2}
    assert is_probable_prime(299996953) and not is_probable_prime(299996953 * 3)


def test_sample_passes_validation(sample):
    report = validate(sample)
    assert report["curves"] == 21
    assert not any(report["failures"].values()), report


def _corrupt(sample, label, **changes):
    df = sample.copy()
    i = df.index[df.label == label][0]
    for col, value in changes.items():
        df.loc[i, col] = value
    return validate(df)["failures"]


@pytest.mark.parametrize("label, changes, check", [
    ("11.a2", {"a6": "-21"}, "bad_prime_not_in_discriminant"),  # Δ = -3 * 60497
    ("11.a2", {"a6": "-24"}, "discriminant_has_good_prime"),  # Δ = -247995, divisible by 11 and more
    ("11.a2", {"rank": 1}, "rank_varies_in_isogeny_class"),
    ("11.a2", {"torsion": "[11]"}, "torsion_not_in_mazur_list"),
    ("11.a2", {"sha": 2}, "sha_not_a_positive_square"),
    ("11.a2", {"conductor": 13}, "label_conductor_mismatch"),
    ("11.a3", {"a4": "0", "a6": "0", "a1": 0, "a2": 0, "a3": 0}, "singular"),
])
def test_validation_catches_corruption(sample, label, changes, check):
    failures = _corrupt(sample, label, **changes)
    assert failures[check] >= 1


@pytest.mark.data
def test_full_table():
    if not Path(DEFAULT_TABLE).exists():
        pytest.skip("data/curves.parquet not built")
    df = load_curves()
    assert len(df) == 3_824_372
    assert df["rank"].value_counts().sort_index().to_dict() == {
        0: 1_404_510, 1: 1_887_132, 2: 493_291, 3: 37_334, 4: 2_086, 5: 19}
    assert df.iso_class.nunique() == 2_917_287
