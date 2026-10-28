"""Section 5 numbers, tables and figures: gradient boosting and the feature ablation.

    python scripts/figs_step05.py            # trains everything (about 40 minutes)
    python scripts/figs_step05.py --reuse    # rebuilds tables, figures and numbers from results/

Needs the Step 4 predictions in data/predictions/ (run scripts/figs_step04.py first). Writes
results/step05_*.csv, data/predictions/step05_*, and notes/figures/step05_*.
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, roc_auc_score

from eclab.dataset import class_codes, load_frame, task_split
from eclab.evaluate import auc_replicates, interval, report
from eclab.models import Boosting
from eclab.splits import SPLITS, validation_mask

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

GROUPS = {"gb-inv": ("invariants",), "gb-ap": ("ap",), "gb-inv+ap": ("invariants", "ap")}
LEAVES = (15, 63, 255)
SEED = 0


def tune(frame):
    """Task 2, random split: number of leaves per tree, chosen by log-loss on the validation classes."""
    tr, _, ytr, _ = task_split(frame, 2, "random")
    val = validation_mask(tr)
    rows = []
    for leaves in LEAVES:
        t0 = time.perf_counter()
        gb = Boosting("invariants", "ap", max_leaf_nodes=leaves).fit(tr, ytr, val)
        p = gb.predict(tr[val])
        rows.append({"max_leaf_nodes": leaves, "trees": gb.n_trees, "val_log_loss": log_loss(ytr[val], p),
                     "val_roc_auc": roc_auc_score(ytr[val], p), "seconds": time.perf_counter() - t0})
        print(rows[-1])
    return pd.DataFrame(rows)


def run(frame, leaves, task, name):
    tr, te, ytr, yte = task_split(frame, task, name)
    val = validation_mask(tr)
    before = pd.read_parquet(PRED / f"step04_task{task}_{name}.parquet")
    assert (before.iso_class.to_numpy() == te.iso_class.to_numpy()).all()
    preds = {"sum": before["sum"].to_numpy(), "lr-inv+ap": before["inv+ap"].to_numpy()}
    models = {}
    for m, groups in GROUPS.items():
        models[m] = Boosting(*groups, max_leaf_nodes=leaves).fit(tr, ytr, val)
        preds[m] = models[m].predict(te)
    codes = class_codes(te.iso_class)
    out = report(yte, preds, codes, baseline="sum")
    # the two questions of the ablation, as paired differences on the same replicates
    reps = auc_replicates(yte, {k: preds[k] for k in ("lr-inv+ap", "gb-ap", "gb-inv+ap")}, codes)
    extra = {}
    for a, b, key in (("gb-inv+ap", "gb-ap", "inv_adds"), ("gb-inv+ap", "lr-inv+ap", "trees_add")):
        lo, hi = interval(reps[a] - reps[b])
        extra |= {key: roc_auc_score(yte, preds[a]) - roc_auc_score(yte, preds[b]), f"{key}_lo": lo, f"{key}_hi": hi}
    out.insert(0, "split", name)
    out.insert(0, "task", task)
    pd.DataFrame(preds | {"y": yte, "iso_class": te.iso_class.to_numpy()}).to_parquet(
        PRED / f"step05_task{task}_{name}.parquet", index=False)
    return out, extra, models, te, yte


def semistable_auc(te, yte, preds):
    """Task 1 on semistable test curves, where the root number is a function of the invariants."""
    s = te.semistable.to_numpy()
    return {k: roc_auc_score(yte[s], p[s]) for k, p in preds.items()}


def importance(model, te, yte, n=200_000):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(te), size=min(n, len(te)), replace=False)
    X = te[model.columns].to_numpy(np.float32)[idx]
    r = permutation_importance(model.model, X, yte[idx], scoring="roc_auc", n_repeats=3, random_state=SEED)
    return pd.Series(r.importances_mean, index=model.columns).sort_values(ascending=False)


def fig_ablation(m):
    """Task 2: ROC-AUC minus that of S, with paired 95% intervals, for each split."""
    show = ["lr-inv+ap", "gb-ap", "gb-inv+ap"]  # boosting on invariants alone is far off the scale (see table)
    names = {"lr-inv+ap": "logistic, inv + a_p", "gb-ap": "boosting, a_p", "gb-inv+ap": "boosting, inv + a_p"}
    fig, axes = plt.subplots(1, 4, figsize=(6.4, 2.0), sharey=True, sharex=True)
    for ax, name in zip(axes, SPLITS):
        d = m[(m.task == 2) & (m.split == name)].set_index("model").loc[show]
        y = np.arange(len(show))[::-1]
        for k, (yy, r) in enumerate(zip(y, d.itertuples())):
            ax.plot([r.diff_lo, r.diff_hi], [yy, yy], color=SERIES[k], lw=2, solid_capstyle="round")
            ax.plot(r.diff, yy, "o", color=SERIES[k], ms=5, mec="white", mew=1)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(name, fontsize=9, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([names[s] for s in show])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_xlabel("AUC minus AUC of $S$")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG / "step05_ablation.pdf")
    plt.close(fig)


def fig_importance(imp):
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.barh(range(len(top)), top.to_numpy(), color=SERIES[0], height=0.6)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([c.replace("_", " ") for c in top.index], fontsize=8)
    ax.set_xlabel("drop in ROC-AUC when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "step05_importance.pdf")
    plt.close(fig)


def by_conductor(frame, bins=8):
    """S on Task 2 in the complete range, by conductor octile: AUC and the median S of each rank."""
    f = frame[(frame.conductor <= 500_000) & frame["rank"].isin([0, 2])]
    rows = []
    for b, g in f.groupby(pd.qcut(f.conductor, bins), observed=True):
        y, s = (g["rank"] == 2).to_numpy(), g.mn_sum.to_numpy()
        rows.append({"lo": b.left, "hi": b.right, "auc": roc_auc_score(y, s),
                     "median_s_rank0": np.median(s[~y]), "median_s_rank2": np.median(s[y])})
    return pd.DataFrame(rows)


def train(frame):
    tuning = tune(frame)
    leaves = int(tuning.loc[tuning.val_log_loss.idxmin(), "max_leaf_nodes"])
    RESULTS.mkdir(exist_ok=True)
    tuning.to_csv(RESULTS / "step05_tuning.csv", index=False, float_format="%.5f")

    rows, extras, imp = [], [], None
    for task, name in [(2, s) for s in SPLITS] + [(1, "random")]:
        t0 = time.perf_counter()
        out, extra, models, te, yte = run(frame, leaves, task, name)
        rows.append(out)
        extras.append({"task": task, "split": name} | extra | {"trees": models["gb-inv+ap"].n_trees})
        if (task, name) == (2, "random"):
            imp = importance(models["gb-inv+ap"], te, yte)
        print(f"task {task} {name:9s} {time.perf_counter() - t0:5.0f}s  "
              + "  ".join(f"{r.model}={r.roc_auc:.4f}" for r in out.itertuples()))
    m = pd.concat(rows, ignore_index=True)
    ex = pd.DataFrame(extras)
    m.to_csv(RESULTS / "step05_metrics.csv", index=False, float_format="%.5f")
    ex.to_csv(RESULTS / "step05_ablation.csv", index=False, float_format="%.5f")
    imp.to_csv(RESULTS / "step05_importance.csv", header=["auc_drop"], float_format="%.5f")


def main():
    frame = load_frame()
    if "--reuse" not in sys.argv:
        train(frame)
    tuning = pd.read_csv(RESULTS / "step05_tuning.csv")
    m = pd.read_csv(RESULTS / "step05_metrics.csv")
    ex = pd.read_csv(RESULTS / "step05_ablation.csv")
    imp = pd.read_csv(RESULTS / "step05_importance.csv", index_col=0)["auc_drop"]
    _, te, _, yte = task_split(frame, 1, "random")
    preds = pd.read_parquet(PRED / "step05_task1_random.parquet")
    semi = semistable_auc(te, yte, {k: preds[k].to_numpy() for k in ("lr-inv+ap", "gb-inv")})
    leaves = int(tuning.loc[tuning.val_log_loss.idxmin(), "max_leaf_nodes"])
    octiles = by_conductor(frame)
    octiles.to_csv(RESULTS / "step05_by_conductor.csv", index=False, float_format="%.5f")
    fig_ablation(m)
    fig_importance(imp)

    # table: Task 2 ROC-AUC per split, with the two ablation differences
    lines = []
    for model, label in (("sum", "$S$ only"), ("lr-inv+ap", "logistic, inv $+$ $a_p$"), ("gb-inv", "boosting, invariants"),
                         ("gb-ap", "boosting, $a_p$"), ("gb-inv+ap", "boosting, inv $+$ $a_p$")):
        d = m[(m.task == 2) & (m.model == model)].set_index("split").loc[list(SPLITS)]
        lines.append(f"{label} & " + " & ".join(f"{v:.3f}" for v in d.roc_auc) + r"\\")
    lines.append(r"\midrule")
    for key, label in (("inv_adds", "invariants add"), ("trees_add", "trees add")):
        d = ex[ex.task == 2].set_index("split").loc[list(SPLITS)]
        cells = [f"${r[key]:+.4f}$ {{\\scriptsize$\\pm{(r[key + '_hi'] - r[key + '_lo']) / 2:.4f}$}}" for _, r in d.iterrows()]
        lines.append(f"{label} & " + " & ".join(cells) + r"\\")
    (FIG / "step05_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    g = lambda task, split, model, col: m[(m.task == task) & (m.split == split) & (m.model == model)][col].iloc[0]
    e = lambda split, col: ex[(ex.task == 2) & (ex.split == split)][col].iloc[0]
    t = tuning.set_index("max_leaf_nodes")
    nums = {
        "Leaves": f"{leaves}",
        "LeavesTrees": f"{int(t.loc[leaves, 'trees'])}",
        "TuneLossSmall": f"{t.loc[LEAVES[0], 'val_log_loss']:.4f}",
        "TuneLossMid": f"{t.loc[LEAVES[1], 'val_log_loss']:.4f}",
        "TuneLossBig": f"{t.loc[LEAVES[2], 'val_log_loss']:.4f}",
        "GbBothRandom": f"{g(2, 'random', 'gb-inv+ap', 'roc_auc'):.3f}",
        "GbBothRandomLo": f"{g(2, 'random', 'gb-inv+ap', 'roc_lo'):.3f}",
        "GbBothRandomHi": f"{g(2, 'random', 'gb-inv+ap', 'roc_hi'):.3f}",
        "GbBothLogLoss": f"{g(2, 'random', 'gb-inv+ap', 'log_loss'):.3f}",
        "GbBothPrAuc": f"{g(2, 'random', 'gb-inv+ap', 'pr_auc'):.3f}",
        "GbApRandom": f"{g(2, 'random', 'gb-ap', 'roc_auc'):.3f}",
        "GbInvRandom": f"{g(2, 'random', 'gb-inv', 'roc_auc'):.3f}",
        "InvAddsRandom": f"{e('random', 'inv_adds'):+.3f}",
        "InvAddsPrime": f"{e('prime', 'inv_adds'):+.3f}",
        "InvAddsFamily": f"{e('family', 'inv_adds'):+.3f}",
        "TreesAddRandom": f"{e('random', 'trees_add'):+.3f}",
        "TreesAddPrime": f"{e('prime', 'trees_add'):+.3f}",
        "GbBothPrime": f"{g(2, 'prime', 'gb-inv+ap', 'roc_auc'):.3f}",
        "GbBothFamily": f"{g(2, 'family', 'gb-inv+ap', 'roc_auc'):.3f}",
        "GbBothConductor": f"{g(2, 'conductor', 'gb-inv+ap', 'roc_auc'):.3f}",
        "TaskOneLrBoth": f"{g(1, 'random', 'lr-inv+ap', 'roc_auc'):.3f}",
        "TaskOneGbInv": f"{g(1, 'random', 'gb-inv', 'roc_auc'):.3f}",
        "TaskOneGbBoth": f"{g(1, 'random', 'gb-inv+ap', 'roc_auc'):.3f}",
        "SemiLrBoth": f"{semi['lr-inv+ap']:.3f}",
        "SemiGbInv": f"{semi['gb-inv']:.3f}",
        "TopFeature": imp.index[0].replace("_", r"\_"),
        "TopFeatureDrop": f"{imp.iloc[0]:.3f}",
        "MisorderSum": f"{1 - g(2, 'random', 'sum', 'roc_auc'):.4f}",
        "MisorderGb": f"{1 - g(2, 'random', 'gb-inv+ap', 'roc_auc'):.4f}",
        "MisorderRatio": f"{(1 - g(2, 'random', 'sum', 'roc_auc')) / (1 - g(2, 'random', 'gb-inv+ap', 'roc_auc')):.0f}",
        "LogNDrop": f"{imp['log_conductor']:.3f}",
        "LogNRank": f"{list(imp.index).index('log_conductor') + 1}",
        "OctAucLow": f"{octiles.auc.iloc[0]:.3f}",
        "OctAucHigh": f"{octiles.auc.iloc[-1]:.3f}",
        "OctMedTwoLow": f"{octiles.median_s_rank2.iloc[0]:.2f}",
        "OctMedTwoHigh": f"{octiles.median_s_rank2.iloc[-1]:.2f}",
        "OctMedZeroLow": f"{octiles.median_s_rank0.iloc[0]:.2f}",
        "OctMedZeroHigh": f"{octiles.median_s_rank0.iloc[-1]:.2f}",
        "OctHiFirst": f"{int(octiles.hi.iloc[0]):,}",
        "OctLoLast": f"{int(octiles.lo.iloc[-1]):,}",
    }
    lines = ["% generated by scripts/figs_step05.py; do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in nums.items()]
    (FIG / "step05_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(tuning.to_string(index=False))
    print(imp.head(12).to_string())
    for k, v in nums.items():
        print(f"{k:18s} {v}")


if __name__ == "__main__":
    main()
