r"""
make_report_figures.py: turns the CSVs from analyze_results_extra.py into the figures and
LaTeX tables used in the Results chapter. No retraining, runs in seconds on your PC.

Usage (from the folder that contains report_tables/):
    python make_report_figures.py --tables_dir report_tables --out_dir figures

Writes into --out_dir:
    confusion_matrices.pdf / .png   row normalised confusion matrices, one panel per condition
    training_curves.pdf / .png      mean validation loss and validation macro F1 by epoch
    per_class_recall_table.tex      \input{} this inside a table environment
    paired_tests_table.tex
    bootstrap_table.tex
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDS = ["none", "handdesigned", "fci"]
NICE = {"none": "No graph", "handdesigned": "Hand designed", "fci": "FCI learned"}
CLASSES = ["happiness", "others", "disgust", "surprise", "repression"]


def fmt_p(p):
    if pd.isna(p):
        return "n/a"
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def write_rows(path, header, rows, colspec):
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def confusion_figure(tdir, odir):
    present = [c for c in CONDS if os.path.exists(os.path.join(tdir, f"confusion_{c}.csv"))]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 4.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, present):
        cm = pd.read_csv(os.path.join(tdir, f"confusion_{c}.csv"), index_col=0).to_numpy().astype(float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right")
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_title(NICE[c])
    axes[0].set_ylabel("True")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(odir, f"confusion_matrices.{ext}"), dpi=300)
    plt.close(fig)


def curves_figure(tdir, odir, min_folds):
    path = os.path.join(tdir, "epoch_curves.csv")
    if not os.path.exists(path):
        return
    cv = pd.read_csv(path)
    cv = cv[cv["n_folds_still_running"] >= min_folds]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for c in CONDS:
        g = cv[cv["condition"] == c]
        if g.empty:
            continue
        axes[0].plot(g["epoch"], g["val_loss"], marker="o", ms=3, label=NICE[c])
        axes[1].plot(g["epoch"], g["val_macro_f1"], marker="o", ms=3, label=NICE[c])
    axes[0].set_ylabel("Validation loss")
    axes[1].set_ylabel("Validation macro F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(odir, f"training_curves.{ext}"), dpi=300)
    plt.close(fig)


def tables(tdir, odir):
    p = os.path.join(tdir, "per_class_recall.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        sup = d[d["condition"] == CONDS[0]].set_index("class")["support"]
        rows = []
        for cl in CLASSES:
            r = [cl.capitalize(), str(int(sup[cl]))]
            for c in CONDS:
                v = d[(d["condition"] == c) & (d["class"] == cl)]["recall"]
                r.append(f"{v.iloc[0]:.3f}" if len(v) else "n/a")
            rows.append(r)
        write_rows(os.path.join(odir, "per_class_recall_table.tex"),
                   ["Class", "Sequences"] + [NICE[c] for c in CONDS], rows, "lrrrr")

    p = os.path.join(tdir, "paired_tests.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        names = {"acc": "Accuracy", "macro_f1": "Macro F1"}
        rows = [[f"{NICE[r.cond_b]} vs {NICE[r.cond_a]}", names[r.metric],
                 f"{r.mean_diff_b_minus_a:+.3f}", f"{int(r.folds_b_better)} / {int(r.folds_a_better)}",
                 fmt_p(r.wilcoxon_p)] for r in d.itertuples()]
        write_rows(os.path.join(odir, "paired_tests_table.tex"),
                   ["Comparison", "Metric", "Mean difference", "Folds better (first named / second named)", "Wilcoxon $p$"],
                   rows, "llrcr")

    p = os.path.join(tdir, "bootstrap_ci.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        def label(item):
            if " - " in item:
                b, a = item.split(" - ")
                return f"{NICE[b]} minus {NICE[a]}"
            return NICE[item]
        rows = [[label(r.item), r.metric.upper(), f"{r.mean:.3f}", f"[{r.ci_low:.3f}, {r.ci_high:.3f}]"]
                for r in d.itertuples()]
        write_rows(os.path.join(odir, "bootstrap_table.tex"),
                   ["Quantity", "Metric", "Mean", "95\\% interval"], rows, "llrr")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables_dir", default="report_tables")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--min_folds", type=int, default=13,
                    help="Only plot epochs where at least this many folds were still training")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    confusion_figure(a.tables_dir, a.out_dir)
    curves_figure(a.tables_dir, a.out_dir, a.min_folds)
    tables(a.tables_dir, a.out_dir)
    print("Wrote:", ", ".join(sorted(os.listdir(a.out_dir))))
