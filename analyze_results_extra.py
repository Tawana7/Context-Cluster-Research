"""
analyze_results_extra.py — analysis-only add-on for the report. NO retraining, no GPU.

Reads the CSVs train_cimer_casme2.py already wrote:
    <prefix><cond>.csv               per-fold rows
    <prefix><cond>_predictions.csv   per-sample predictions
    <prefix><cond>_epochs.csv        per-fold x epoch curves
and writes report-ready tables into --out_dir:
    paired_tests.csv        Wilcoxon signed-rank on per-fold acc / macro-F1 (folds are matched)
    bootstrap_ci.csv        subject-level bootstrap 95% CIs for pooled UAR / UF1, and for
                            the DIFFERENCE between conditions (resampling subjects, paired)
    per_class_recall.csv    recall per emotion class per condition
    confusion_<cond>.csv    pooled confusion matrix per condition
    epoch_curves.csv        mean across folds of train/val/test metrics by epoch

Usage:
    python analyze_results_extra.py --results_dir . --prefix results_v2_ --out_dir report_tables
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

CONDITIONS = ["none", "handdesigned", "fci"]
CLASS_NAMES = ["happiness", "others", "disgust", "surprise", "repression"]  # = CASME2_LABEL_TO_CLS order


def load(results_dir, prefix, cond, suffix):
    path = os.path.join(results_dir, f"{prefix}{cond}{suffix}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def uar_uf1(y, p):
    present = sorted(set(y.tolist()))
    return (balanced_accuracy_score(y, p),
            f1_score(y, p, labels=present, average="macro", zero_division=0))


def paired_tests(folds):
    rows = []
    for a, b in itertools.combinations([c for c in CONDITIONS if c in folds], 2):
        da, db = folds[a].set_index("fold"), folds[b].set_index("fold")
        common = da.index.intersection(db.index)
        for m in ("acc", "macro_f1"):
            x, y = da.loc[common, m].to_numpy(), db.loc[common, m].to_numpy()
            diff = y - x
            p = np.nan if np.allclose(diff, 0) else wilcoxon(x, y).pvalue
            rows.append({"cond_a": a, "cond_b": b, "metric": m, "n_folds": len(common),
                         "mean_a": x.mean(), "mean_b": y.mean(), "mean_diff_b_minus_a": diff.mean(),
                         "folds_b_better": int((diff > 0).sum()), "folds_a_better": int((diff < 0).sum()),
                         "wilcoxon_p": p})
    return pd.DataFrame(rows)


def bootstrap(preds, n_boot=2000, seed=0):
    """Resample SUBJECTS (the unit LOSO generalises over), same resample for every condition."""
    conds = [c for c in CONDITIONS if c in preds]
    subs = sorted(set.intersection(*[set(preds[c]["held_out_subject"]) for c in conds]))
    by_sub = {c: {s: g for s, g in preds[c].groupby("held_out_subject")} for c in conds}
    rng = np.random.default_rng(seed)
    draws = {c: {"uar": [], "uf1": []} for c in conds}
    for _ in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        for c in conds:
            g = pd.concat([by_sub[c][s] for s in pick])
            u, f = uar_uf1(g["label"].to_numpy(), g["pred"].to_numpy())
            draws[c]["uar"].append(u)
            draws[c]["uf1"].append(f)
    rows = []
    for c in conds:
        for m in ("uar", "uf1"):
            d = np.array(draws[c][m])
            rows.append({"item": c, "metric": m, "mean": d.mean(),
                         "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)})
    for a, b in itertools.combinations(conds, 2):
        for m in ("uar", "uf1"):
            d = np.array(draws[b][m]) - np.array(draws[a][m])
            rows.append({"item": f"{b} - {a}", "metric": m, "mean": d.mean(),
                         "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=".")
    ap.add_argument("--prefix", default="results_v2_")
    ap.add_argument("--out_dir", default="report_tables")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    folds = {c: load(a.results_dir, a.prefix, c, "") for c in CONDITIONS}
    preds = {c: load(a.results_dir, a.prefix, c, "_predictions") for c in CONDITIONS}
    epochs = {c: load(a.results_dir, a.prefix, c, "_epochs") for c in CONDITIONS}
    folds = {c: d for c, d in folds.items() if d is not None}
    preds = {c: d for c, d in preds.items() if d is not None}
    epochs = {c: d for c, d in epochs.items() if d is not None}

    pt = paired_tests(folds)
    pt.to_csv(os.path.join(a.out_dir, "paired_tests.csv"), index=False)
    print("\nPaired Wilcoxon (per-fold):\n", pt.round(4).to_string(index=False))

    bs = bootstrap(preds, a.n_boot)
    bs.to_csv(os.path.join(a.out_dir, "bootstrap_ci.csv"), index=False)
    print("\nSubject-level bootstrap (95% CI):\n", bs.round(4).to_string(index=False))

    rec_rows = []
    for c, p in preds.items():
        y, yh = p["label"].to_numpy(), p["pred"].to_numpy()
        rec = recall_score(y, yh, labels=range(len(CLASS_NAMES)), average=None, zero_division=0)
        support = np.bincount(y, minlength=len(CLASS_NAMES))
        for i, name in enumerate(CLASS_NAMES):
            rec_rows.append({"condition": c, "class": name, "support": int(support[i]), "recall": rec[i]})
        cm = confusion_matrix(y, yh, labels=range(len(CLASS_NAMES)))
        pd.DataFrame(cm, index=[f"true_{n}" for n in CLASS_NAMES],
                     columns=[f"pred_{n}" for n in CLASS_NAMES]).to_csv(
            os.path.join(a.out_dir, f"confusion_{c}.csv"))
    pc = pd.DataFrame(rec_rows)
    pc.to_csv(os.path.join(a.out_dir, "per_class_recall.csv"), index=False)
    print("\nPer-class recall:\n", pc.pivot(index="class", columns="condition", values="recall").round(3))

    cols = ["train_acc", "val_loss", "val_acc", "val_macro_f1", "test_acc"]
    curves = []
    for c, e in epochs.items():
        have = [k for k in cols if k in e.columns]
        g = e.groupby("epoch")[have].mean()
        g["n_folds_still_running"] = e.groupby("epoch").size()
        g.insert(0, "condition", c)
        curves.append(g.reset_index())
    if curves:
        cv = pd.concat(curves)
        cv.to_csv(os.path.join(a.out_dir, "epoch_curves.csv"), index=False)
        print("\nEpoch curves (mean over folds still running; test_acc is DIAGNOSTIC ONLY):")
        print(cv[cv["epoch"] <= 16].round(3).to_string(index=False))

    print(f"\nTables written to {a.out_dir}/")


if __name__ == "__main__":
    main()
