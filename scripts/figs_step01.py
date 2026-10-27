"""Section 1 numbers and figures: what is in the data, and the two conductor regimes.

    python scripts/figs_step01.py

Reads data/curves.parquet and data/validation.json (from `python -m eclab.data.build`) and
writes notes/figures/step01_numbers.tex and two figures.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eclab.curves import is_probable_prime
from eclab.data.ecdata import load_curves

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "notes" / "figures"
COMPLETE_UP_TO = 500_000

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, validated
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # one-hue ramp for ranks 0,1,2,3+

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def regime(df):
    """complete (N <= 500k), prime (N > 500k prime) or smooth (N > 500k, 7-smooth)."""
    hi = df.conductor > COMPLETE_UP_TO
    prime_n = {n: is_probable_prime(int(n)) for n in df.conductor[hi].unique()}
    out = np.where(hi, np.where(df.conductor.map(prime_n).fillna(False).astype(bool), "prime", "smooth"), "complete")
    if not (df.conductor[out == "smooth"].map(lambda n: _is_7_smooth(int(n)))).all():
        raise ValueError("a curve above 500k is neither prime nor 7-smooth")
    return pd.Series(out, index=df.index)


def _is_7_smooth(n):
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1


def rank_shares(ranks):
    r = np.minimum(ranks, 3)
    return np.bincount(r, minlength=4) / len(r)


def fig_rank_by_conductor(df):
    """Share of each rank by conductor, in the complete range only (bins of 25,000)."""
    d = df[df.conductor <= COMPLETE_UP_TO]
    edges = np.arange(0, COMPLETE_UP_TO + 1, 25_000)
    b = np.digitize(d.conductor, edges[1:-1])
    r = np.minimum(d["rank"].to_numpy(), 3)
    shares = np.array([np.bincount(r[b == i], minlength=4) / (b == i).sum() for i in range(len(edges) - 1)])
    mid = (edges[:-1] + edges[1:]) / 2 / 1000

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for k, name in enumerate(["rank 0", "rank 1", "rank 2", "rank ≥ 3"]):
        ax.plot(mid, shares[:, k], color=SERIES[k], lw=2, marker="o", ms=3)
        ax.annotate(name, (mid[-1], shares[-1, k]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=8)
    ax.set_xlim(0, 530)
    ax.set_ylim(0, None)
    ax.set_xlabel("conductor N (thousands), bins of 25,000")
    ax.set_ylabel("share of curves in bin")
    fig.tight_layout()
    fig.savefig(FIG / "step01_rank_by_conductor.pdf")
    plt.close(fig)
    return shares


def fig_regimes(groups):
    """100% stacked bars: rank shares per regime."""
    fig, ax = plt.subplots(figsize=(6.2, 2.3))
    names = list(groups)
    left = np.zeros(len(names))
    for k, lab in enumerate(["0", "1", "2", "≥ 3"]):
        w = np.array([groups[n][k] for n in names])
        ax.barh(names, w, left=left, color=ORDINAL[k], edgecolor="white", linewidth=1.5, label=f"rank {lab}")
        for y, (l, v) in enumerate(zip(left, w)):
            if v > 0.06:
                ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if k >= 1 else INK)
        left += w
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "step01_regimes.pdf")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_curves()
    report = json.loads((ROOT / "data" / "validation.json").read_text())
    reg = regime(df)
    complete = reg == "complete"
    small_prime = complete & df.conductor.map(lambda n: is_probable_prime(int(n)))

    groups = {
        f"all, N ≤ 500k ({complete.sum():,})": rank_shares(df["rank"][complete].to_numpy()),
        f"prime N ≤ 500k ({small_prime.sum():,})": rank_shares(df["rank"][small_prime].to_numpy()),
        f"prime N > 500k ({(reg == 'prime').sum():,})": rank_shares(df["rank"][reg == "prime"].to_numpy()),
        f"7-smooth N > 500k ({(reg == 'smooth').sum():,})": rank_shares(df["rank"][reg == "smooth"].to_numpy()),
    }
    fig_regimes(groups)
    shares = fig_rank_by_conductor(df)

    t2 = df["rank"].isin([0, 2])
    pi = lambda m: (df["rank"][m & t2] == 2).mean()
    big = df.a4.map(lambda s: abs(int(s)) >= 2**63) | df.a6.map(lambda s: abs(int(s)) >= 2**63)
    class_size = df.groupby("iso_class").size()
    sha_gt1 = lambda r: (df.sha[df["rank"] == r] > 1).mean()

    def pct(x):
        return f"{100 * x:.1f}\\%"

    nums = {
        "NCurves": f"{len(df):,}",
        "NClasses": f"{df.iso_class.nunique():,}",
        "NSingletonClasses": pct((class_size == 1).mean()),
        "MaxClassSize": f"{class_size.max()}",
        "NComplete": f"{complete.sum():,}",
        "NHigh": f"{(~complete).sum():,}",
        "NHighPrime": f"{(reg == 'prime').sum():,}",
        "NHighSmooth": f"{(reg == 'smooth').sum():,}",
        "NSmallPrime": f"{small_prime.sum():,}",
        "MaxConductor": f"{df.conductor.max():,}",
        "NBigCoeffs": f"{big.sum():,}",
        "NChecks": f"{len(report['failures'])}",
        "NFailures": f"{sum(report['failures'].values()):,}",
        "NRankZero": f"{(df['rank'] == 0).sum():,}",
        "NRankOne": f"{(df['rank'] == 1).sum():,}",
        "NRankTwo": f"{(df['rank'] == 2).sum():,}",
        "NRankThree": f"{(df['rank'] == 3).sum():,}",
        "NRankFour": f"{(df['rank'] == 4).sum():,}",
        "NRankFive": f"{(df['rank'] == 5).sum():,}",
        "NTaskTwo": f"{t2.sum():,}",
        "PiTaskTwo": pct(pi(pd.Series(True, index=df.index))),
        "PiTaskTwoComplete": pct(pi(complete)),
        "PiTaskTwoHighPrime": pct(pi(reg == "prime")),
        "PiTaskTwoSmallPrime": pct(pi(small_prime)),
        "RankTwoFirstBin": pct(shares[0, 2]),
        "RankTwoLastBin": pct(shares[-1, 2]),
        "RankZeroFirstBin": pct(shares[0, 0]),
        "RankZeroLastBin": pct(shares[-1, 0]),
        "ShaBigRankZero": pct(sha_gt1(0)),
        "ShaBigRankTwo": pct(sha_gt1(2)),
    }
    lines = ["% generated by scripts/figs_step01.py; do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in nums.items()]
    (FIG / "step01_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, v in nums.items():
        print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()
