"""
au_features.py — Turn CASME II's "Action Units" column into AU activation
vectors, and discover the AU vocabulary directly from your data rather than
assuming a fixed list (CASME II subsets vary slightly in which AUs appear).

CASME II's coding sheet writes multiple co-active AUs per sample like
"4+7", "4,7", or occasionally with a laterality prefix like "L4+R12".
This module normalizes those into a fixed-order binary vector.

NOTE ON DATA TYPE: CASME II's public coding sheet gives AU *presence*,
not intensity — so these are binary (0/1) vectors, not continuous
activation values. FCI's default Fisher's Z independence test assumes
continuous data; see fci_graph_learning.py for the workaround options.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd


def parse_au_string(au_string):
    """
    '4+7' -> ['4', '7']
    'L4+R12' -> ['4', '12']   (laterality dropped — CASME II doesn't
                               consistently distinguish L/R across all
                               subjects, and the hand-designed graph
                               below is defined at the AU-number level)
    '/' or NaN -> []          (no AUs coded, e.g. some 'others' rows)
    """
    if pd.isna(au_string) or str(au_string).strip() in ("", "/"):
        return []
    parts = re.split(r"[+,]", str(au_string))
    aus = []
    for p in parts:
        m = re.search(r"(\d+)", p)
        if m:
            aus.append(m.group(1))
    return aus


def discover_au_vocabulary(xlsx_path, min_count=2):
    """
    Scan the whole coding sheet and return the sorted list of AU codes
    that appear at least `min_count` times. Rare AUs (appearing once)
    are dropped — FCI can't learn anything meaningful about a variable
    with near-zero variance, and a graph node with one observation isn't
    a usable node either way.
    """
    df = pd.read_excel(xlsx_path)
    counts = Counter()
    for entry in df["Action Units"]:
        counts.update(parse_au_string(entry))
    vocab = sorted(
        [au for au, c in counts.items() if c >= min_count],
        key=lambda x: int(x),
    )
    return vocab, counts


def au_vector(au_string, vocab):
    """Binary presence vector, ordered per `vocab`."""
    present = set(parse_au_string(au_string))
    return np.array([1.0 if au in present else 0.0 for au in vocab])


def build_au_matrix(xlsx_path, vocab):
    """
    Returns (X, subjects, labels) where X is [N, len(vocab)] binary AU
    matrix — the direct input to FCI, and the thing the graph-gated
    CI-Block's node dimensions are aligned to.
    """
    df = pd.read_excel(xlsx_path)
    X = np.stack([au_vector(a, vocab) for a in df["Action Units"]])
    subjects = df["Subject"].to_numpy()
    labels = df["Estimated Emotion"].to_numpy()
    return X, subjects, labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--min_count", type=int, default=2)
    args = parser.parse_args()

    vocab, counts = discover_au_vocabulary(args.xlsx, args.min_count)
    print(f"Discovered {len(vocab)} AU nodes (min_count={args.min_count}):")
    for au in vocab:
        print(f"  AU{au}: {counts[au]} occurrences")

    X, subjects, labels = build_au_matrix(args.xlsx, vocab)
    print(f"\nAU matrix shape: {X.shape}")
    print(f"Mean activation rate per AU: {X.mean(axis=0).round(3)}")
