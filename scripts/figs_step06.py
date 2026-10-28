"""Section 6 numbers, tables and figures: a small neural network, and calibration.

    python scripts/figs_step06.py            # trains the networks (about 15 minutes)
    python scripts/figs_step06.py --reuse    # rebuilds tables, figures and numbers from results/

Needs the Step 5 predictions in data/predictions/ (run scripts/figs_step05.py first). Writes
results/step06_*.csv, data/predictions/step06_*, and notes/figures/step06_*.
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from eclab.dataset import class_codes, load_frame, task_split
from eclab.evaluate import EPS, auc_replicates, ece, estimate_prior, interval, reliability, report, shift_prior
from eclab.models import MLP
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

NETS = {"mlp-ap": ("ap",), "mlp-inv+ap": ("invariants", "ap")}
PAIRS = (("mlp-ap", "gb-ap"), ("mlp-inv+ap", "gb-inv+ap"))


def run(frame, name):
    tr, te, ytr, yte = task_split(frame, 2, name)
    val = validation_mask(tr)
    before = pd.read_parquet(PRED / f"step05_task2_{name}.parquet")
    assert (before.iso_class.to_numpy() == te.iso_class.to_numpy()).all()
    preds = {k: before[k].to_numpy() for k in ("sum", "lr-inv+ap", "gb-ap", "gb-inv+ap")}
    epochs = {}
    for m, groups in NETS.items():
        net = MLP(*groups).fit(tr, ytr, val)
        preds[m] = net.predict(te).astype(np.float64)
        epochs[m] = (net.best_epoch, len(net.history))
    codes = class_codes(te.iso_class)
    out = report(yte, preds, codes, baseline="gb-inv+ap")
    out.insert(0, "split", name)
    reps = auc_replicates(yte, preds, codes)
    pairs = []
    for a, b in PAIRS:
        lo, hi = interval(reps[a] - reps[b])
        pairs.append({"split": name, "net": a, "tree": b, "diff": roc_auc_score(yte, preds[a]) - roc_auc_score(yte, preds[b]),
                      "lo": lo, "hi": hi, "best_epoch": epochs[a][0], "epochs_run": epochs[a][1]})
    pd.DataFrame(preds | {"y": yte, "iso_class": te.iso_class.to_numpy()}).to_parquet(
        PRED / f"step06_task2_{name}.parquet", index=False)
    return out, pd.DataFrame(pairs), ytr.mean()


def label_shift(pi_train):
    """Family split, boosting on inv + a_p: calibration before and after correcting the base rate."""
    d = pd.read_parquet(PRED / "step06_task2_family.parquet")
    y, p = d.y.to_numpy(), np.clip(d["gb-inv+ap"].to_numpy(), EPS, 1 - EPS)
    pi_hat = estimate_prior(p, pi_train)
    fixed = {"raw": p, "oracle": shift_prior(p, pi_train, y.mean()), "em": shift_prior(p, pi_train, pi_hat)}
    out = {"pi_train": pi_train, "pi_test": y.mean(), "pi_mean_pred": p.mean(), "pi_em": pi_hat}
    for k, q in fixed.items():
        out[f"ece_{k}"], out[f"log_loss_{k}"] = ece(y, q), log_loss(y, q)
    return out, y, fixed


def by_log_conductor(frame, bins=5):
    """Family split: ROC-AUC of boosting and of the network (both on inv + a_p) by quintile of log N."""
    _, te, _, y = task_split(frame, 2, "family")
    d = pd.read_parquet(PRED / "step06_task2_family.parquet")
    q = pd.qcut(te.log_conductor.to_numpy(), bins)
    rows = []
    for c in q.categories:
        m = np.asarray(q == c)
        rows.append({"lo": c.left, "hi": c.right, "gb": roc_auc_score(y[m], d["gb-inv+ap"].to_numpy()[m]),
                     "mlp": roc_auc_score(y[m], d["mlp-inv+ap"].to_numpy()[m])})
    return pd.DataFrame(rows)


def fig_calibration(shift_y, shift_fixed):
    d = pd.read_parquet(PRED / "step06_task2_random.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    ax = axes[0]
    for k, (m, label) in enumerate((("lr-inv+ap", "logistic"), ("gb-inv+ap", "boosting"), ("mlp-inv+ap", "network"))):
        r = reliability(d.y.to_numpy(), d[m].to_numpy())
        r = r[r.n >= 200]
        ax.plot(r.p, r.y, color=SERIES[k], lw=2, marker="o", ms=3, label=label)
    ax.set_title("random split", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1]
    for k, (m, label) in enumerate((("raw", "as trained"), ("em", "base rate re-estimated"))):
        r = reliability(shift_y, shift_fixed[m])
        r = r[r.n >= 200]
        ax.plot(r.p, r.y, color=SERIES[k], lw=2, marker="o", ms=3, label=label)
    ax.set_title("family split, boosting", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes:
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted P(rank 2)")
    axes[0].set_ylabel("observed share of rank 2")
    fig.tight_layout()
    fig.savefig(FIG / "step06_calibration.pdf")
    plt.close(fig)


def train(frame):
    rows, pairs = [], []
    for name in SPLITS:
        t0 = time.perf_counter()
        out, pr, _ = run(frame, name)
        rows.append(out)
        pairs.append(pr)
        print(f"{name:9s} {time.perf_counter() - t0:5.0f}s  "
              + "  ".join(f"{r.model}={r.roc_auc:.4f}" for r in out.itertuples()))
    RESULTS.mkdir(exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(RESULTS / "step06_metrics.csv", index=False, float_format="%.5f")
    pd.concat(pairs, ignore_index=True).to_csv(RESULTS / "step06_pairs.csv", index=False, float_format="%.5f")


def main():
    frame = load_frame()
    if "--reuse" not in sys.argv:
        train(frame)
    m = pd.read_csv(RESULTS / "step06_metrics.csv")
    pr = pd.read_csv(RESULTS / "step06_pairs.csv")
    shift, shift_y, shift_fixed = label_shift(task_split(frame, 2, "family")[2].mean())
    quint = by_log_conductor(frame)
    quint.to_csv(RESULTS / "step06_family_by_log_conductor.csv", index=False, float_format="%.5f")
    pd.Series(shift).to_csv(RESULTS / "step06_label_shift.csv", header=["value"], float_format="%.5f")
    fig_calibration(shift_y, shift_fixed)

    # table: the paired network-minus-tree differences, per split
    lines = []
    for net, label in (("mlp-ap", "$a_p$ only"), ("mlp-inv+ap", "invariants $+$ $a_p$")):
        d = pr[pr.net == net].set_index("split").loc[list(SPLITS)]
        cells = [f"${r.diff:+.4f}$ {{\\scriptsize$\\pm{(r.hi - r.lo) / 2:.4f}$}}" for r in d.itertuples()]
        lines.append(f"{label} & " + " & ".join(cells) + r"\\")
    (FIG / "step06_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # table: calibration on the random split
    r = m[m.split == "random"].set_index("model")
    lines = []
    for model, label in (("sum", "logistic on $S$"), ("lr-inv+ap", "logistic, inv $+$ $a_p$"),
                         ("gb-inv+ap", "boosting, inv $+$ $a_p$"), ("mlp-inv+ap", "network, inv $+$ $a_p$")):
        x = r.loc[model]
        lines.append(f"{label} & {x.roc_auc:.4f} & {x.log_loss:.4f} & {x.ece:.4f}\\\\")
    (FIG / "step06_calib_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    g = lambda split, model, col: m[(m.split == split) & (m.model == model)][col].iloc[0]
    q = lambda split, net, col: pr[(pr.split == split) & (pr.net == net)][col].iloc[0]
    nums = {
        "MlpBothFamily": f"{g('family', 'mlp-inv+ap', 'roc_auc'):.3f}",
        "MlpApFamily": f"{g('family', 'mlp-ap', 'roc_auc'):.3f}",
        "MlpBothPrime": f"{g('prime', 'mlp-inv+ap', 'roc_auc'):.3f}",
        "QuintGbFirst": f"{quint.gb.iloc[0]:.3f}",
        "QuintGbLast": f"{quint.gb.iloc[-1]:.3f}",
        "QuintMlpFirst": f"{quint.mlp.iloc[0]:.3f}",
        "QuintMlpLast": f"{quint.mlp.iloc[-1]:.3f}",
        "QuintLastLo": f"{quint.lo.iloc[-1]:.1f}",
        "QuintFirstHi": f"{quint.hi.iloc[0]:.1f}",
        "MlpLogLoss": f"{g('random', 'mlp-inv+ap', 'log_loss'):.3f}",
        "MlpBothRandom": f"{g('random', 'mlp-inv+ap', 'roc_auc'):.3f}",
        "MlpApRandom": f"{g('random', 'mlp-ap', 'roc_auc'):.3f}",
        "MlpDiffRandom": f"{q('random', 'mlp-inv+ap', 'diff'):+.4f}",
        "MlpBestEpoch": f"{q('random', 'mlp-inv+ap', 'best_epoch')}",
        "MlpEpochsRun": f"{q('random', 'mlp-inv+ap', 'epochs_run')}",
        "EceSum": f"{g('random', 'sum', 'ece'):.4f}",
        "EceGb": f"{g('random', 'gb-inv+ap', 'ece'):.4f}",
        "EceMlp": f"{g('random', 'mlp-inv+ap', 'ece'):.4f}",
        "ShiftPiTrain": f"{100 * shift['pi_train']:.1f}\\%",
        "ShiftPiTest": f"{100 * shift['pi_test']:.1f}\\%",
        "ShiftPiMeanPred": f"{100 * shift['pi_mean_pred']:.1f}\\%",
        "ShiftPiEm": f"{100 * shift['pi_em']:.1f}\\%",
        "ShiftEceRaw": f"{shift['ece_raw']:.3f}",
        "ShiftEceOracle": f"{shift['ece_oracle']:.3f}",
        "ShiftEceEm": f"{shift['ece_em']:.3f}",
        "ShiftLossRaw": f"{shift['log_loss_raw']:.4f}",
        "ShiftLossEm": f"{shift['log_loss_em']:.4f}",
    }
    lines = ["% generated by scripts/figs_step06.py; do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in nums.items()]
    (FIG / "step06_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(pr.to_string(index=False))
    print(shift)
    for k, v in nums.items():
        print(f"{k:18s} {v}")


if __name__ == "__main__":
    main()
