"""Section 4 numbers, tables and figures: baselines and logistic regression on both tasks.

    python scripts/figs_step04.py

Reads data/curves.parquet and data/features.parquet. Writes results/step04_metrics.csv,
test-set predictions to data/predictions/, and notes/figures/step04_*.
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eclab.dataset import TASKS, class_codes, load_frame, task_split
from eclab.evaluate import report
from eclab.features import AP_COLUMNS, PRIMES, feature_columns
from eclab.models import Linear, Majority
from eclab.splits import SPLITS

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "notes" / "figures"
RESULTS = ROOT / "results"
PRED = ROOT / "data" / "predictions"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, validated

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})

MODELS = {
    "majority": lambda: Majority(),
    "sum": lambda: Linear("sum"),
    "inv": lambda: Linear("invariants"),
    "ap": lambda: Linear("ap"),
    "inv+ap": lambda: Linear("invariants", "ap"),
}
LABELS = {"majority": "majority", "sum": "$S$ only", "inv": "invariants", "ap": "$a_p$", "inv+ap": "invariants $+$ $a_p$"}


def run(frame):
    rows, weights = [], None
    PRED.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        for name in SPLITS:
            t0 = time.perf_counter()
            tr, te, ytr, yte = task_split(frame, task, name)
            preds = {}
            for m, make in MODELS.items():
                model = make().fit(tr, ytr)
                preds[m] = model.predict(te)
                if (task, name, m) == (2, "random", "ap"):
                    weights = model.coefficients()
            out = report(yte, preds, class_codes(te.iso_class), baseline="sum")
            out.insert(0, "split", name)
            out.insert(0, "task", task)
            out["n_test"], out["pi_test"] = len(yte), yte.mean()
            rows.append(out)
            pd.DataFrame(preds | {"y": yte, "iso_class": te.iso_class.to_numpy()}).to_parquet(
                PRED / f"step04_task{task}_{name}.parquet", index=False)
            print(f"task {task} {name:9s} {time.perf_counter() - t0:5.0f}s  "
                  + "  ".join(f"{r.model}={r.roc_auc:.4f}" for r in out.itertuples()))
    return pd.concat(rows, ignore_index=True), weights


def parity_check(frame):
    """Task 1, random split: what happens if the root number w = (-1)^rank were allowed."""
    tr, te, ytr, yte = task_split(frame, 1, "random")
    w = lambda f: (1 - 2 * (f["rank"].to_numpy() % 2)).reshape(-1, 1)
    alone = roc_auc_score(yte, -w(te).ravel())
    cols = feature_columns("invariants", "ap")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(np.hstack([tr[cols].to_numpy(np.float64), w(tr)]), ytr)
    with_w = roc_auc_score(yte, model.predict_proba(np.hstack([te[cols].to_numpy(np.float64), w(te)]))[:, 1])
    return alone, with_w


def sign_flip(frame):
    """Task 1, family split: the logistic weight on n_split, and n_split as a score on each side."""
    tr, te, ytr, yte = task_split(frame, 1, "family")
    coef = Linear("invariants").fit(tr, ytr).coefficients()["n_split"]
    return coef, roc_auc_score(ytr, tr.n_split), roc_auc_score(yte, te.n_split)


def fig_auc(m):
    """Task 2: ROC-AUC with 95% class-bootstrap intervals, one row per split."""
    show = ["sum", "inv", "ap", "inv+ap"]
    fig, axes = plt.subplots(1, 4, figsize=(6.4, 2.3), sharey=True)
    for ax, name in zip(axes, SPLITS):
        d = m[(m.task == 2) & (m.split == name)].set_index("model").loc[show]
        y = np.arange(len(show))[::-1]
        for k, (yy, r) in enumerate(zip(y, d.itertuples())):
            ax.plot([r.roc_lo, r.roc_hi], [yy, yy], color=SERIES[k], lw=2, solid_capstyle="round")
            ax.plot(r.roc_auc, yy, "o", color=SERIES[k], ms=5, mec="white", mew=1)
        ax.set_title(name, fontsize=9, color=INK)
        ax.set_xlim(0.5, 1.0)
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[s].replace("$", "").replace("\\", "") for s in show])
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("ROC-AUC")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG / "step04_auc_task2.pdf")
    plt.close(fig)


def fig_weights(weights, frame):
    """Learned logistic weights on a_p (Task 2, random split) next to the fixed weights log p / p."""
    sd = frame.loc[frame.conductor <= 500_000, AP_COLUMNS].std().to_numpy()
    raw = -np.array([weights[c] for c in AP_COLUMNS]) / sd  # per unit of a_p; minus: negative a_p means rank 2
    fixed = np.log(PRIMES) / np.array(PRIMES)
    scale = (raw @ fixed) / (fixed @ fixed)  # least-squares match of overall scale
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    ax.plot(PRIMES, fixed, color=SERIES[1], lw=2, marker="o", ms=3)
    ax.plot(PRIMES, raw / scale, color=SERIES[0], lw=2, marker="o", ms=3)
    ax.annotate(r"$\log p/p$, fixed in $S$", (PRIMES[-1], fixed[-1]), xytext=(0, 8), textcoords="offset points",
                color=INK, fontsize=8, ha="right")
    ax.annotate("learned, rescaled", (PRIMES[2], raw[2] / scale), xytext=(10, 6), textcoords="offset points",
                color=INK, fontsize=8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("prime p")
    ax.set_ylabel("weight on $-a_p$")
    fig.tight_layout()
    fig.savefig(FIG / "step04_weights.pdf")
    plt.close(fig)
    big = np.array(PRIMES) >= 7
    slope = np.polyfit(np.log(PRIMES)[big], np.log(raw[big]), 1)[0]  # learned weight ~ p^slope
    return float(np.corrcoef(raw, fixed)[0, 1]), float(slope)


def table(m, task):
    """LaTeX rows: one per model, ROC-AUC on each split, with the paired difference from S below."""
    lines = []
    for model in MODELS:
        d = m[(m.task == task) & (m.model == model)].set_index("split").loc[list(SPLITS)]
        cells = []
        for r in d.itertuples():
            if model in ("majority", "sum"):
                cells.append(f"{r.roc_auc:.3f}")
            else:
                cells.append(f"{r.roc_auc:.3f} {{\\scriptsize({r.diff:+.3f})}}")
        lines.append(f"{LABELS[model]} & " + " & ".join(cells) + r"\\")
    return "\n".join(lines) + "\n"


def main():
    frame = load_frame()
    m, weights = run(frame)
    RESULTS.mkdir(exist_ok=True)
    m.to_csv(RESULTS / "step04_metrics.csv", index=False, float_format="%.5f")
    fig_auc(m)
    corr, slope = fig_weights(weights, frame)
    w_alone, w_with = parity_check(frame)
    flip_coef, flip_train, flip_test = sign_flip(frame)
    (FIG / "step04_table_task1.tex").write_text(table(m, 1), encoding="utf-8")
    (FIG / "step04_table_task2.tex").write_text(table(m, 2), encoding="utf-8")

    g = lambda task, split, model, col: m[(m.task == task) & (m.split == split) & (m.model == model)][col].iloc[0]
    r2 = lambda model, col: g(2, "random", model, col)
    nums = {
        "WeightCorr": f"{corr:.3f}",
        "WeightSlope": f"{slope:.2f}",
        "ParityAloneAuc": f"{w_alone:.3f}",
        "ParityWithAuc": f"{w_with:.3f}",
        "FlipCoef": f"{flip_coef:+.3f}",
        "FlipTrainAuc": f"{flip_train:.3f}",
        "FlipTestAuc": f"{flip_test:.3f}",
        "TaskOneBothAuc": f"{g(1, 'random', 'inv+ap', 'roc_auc'):.3f}",
        "TaskOneInvPrime": f"{g(1, 'prime', 'inv', 'roc_auc'):.3f}",
        "TaskOneInvFamily": f"{g(1, 'family', 'inv', 'roc_auc'):.3f}",
        "TaskOneInvAuc": f"{g(1, 'random', 'inv', 'roc_auc'):.3f}",
        "MajLogLoss": f"{r2('majority', 'log_loss'):.3f}",
        "SumLogLoss": f"{r2('sum', 'log_loss'):.3f}",
        "BothLogLoss": f"{r2('inv+ap', 'log_loss'):.3f}",
        "MajPrAuc": f"{r2('majority', 'pr_auc'):.3f}",
        "SumPrAuc": f"{r2('sum', 'pr_auc'):.3f}",
        "BothPrAuc": f"{r2('inv+ap', 'pr_auc'):.3f}",
        "SumAucRandom": f"{r2('sum', 'roc_auc'):.3f}",
        "BothAucRandom": f"{r2('inv+ap', 'roc_auc'):.3f}",
        "BothDiffRandom": f"{r2('inv+ap', 'diff'):+.4f}",
        "BothDiffLoRandom": f"{r2('inv+ap', 'diff_lo'):+.4f}",
        "BothDiffHiRandom": f"{r2('inv+ap', 'diff_hi'):+.4f}",
        "SumCiLo": f"{r2('sum', 'roc_lo'):.3f}",
        "SumCiHi": f"{r2('sum', 'roc_hi'):.3f}",
        "ApDiffPrime": f"{g(2, 'prime', 'ap', 'diff'):+.3f}",
        "SumAucPrime": f"{g(2, 'prime', 'sum', 'roc_auc'):.3f}",
        "SumAucFamily": f"{g(2, 'family', 'sum', 'roc_auc'):.3f}",
        "BothAucFamily": f"{g(2, 'family', 'inv+ap', 'roc_auc'):.3f}",
        "BothAucPrime": f"{g(2, 'prime', 'inv+ap', 'roc_auc'):.3f}",
        "FamilyBothLogLoss": f"{g(2, 'family', 'inv+ap', 'log_loss'):.2f}",
        "FamilyMaxLogN": f"{frame.loc[frame.conductor_is_prime & (frame.conductor > 500_000), 'log_conductor'].max():.1f}",
        "TrainMaxLogN": f"{np.log(500_000):.1f}",
        "SumEceFamily": f"{g(2, 'family', 'sum', 'ece'):.3f}",
        "SumEceRandom": f"{r2('sum', 'ece'):.3f}",
    }
    lines = ["% generated by scripts/figs_step04.py; do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in nums.items()]
    (FIG / "step04_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, v in nums.items():
        print(f"{k:18s} {v}")


if __name__ == "__main__":
    main()
