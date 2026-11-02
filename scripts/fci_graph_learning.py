"""
fci_graph_learning.py — Learn an AU causal graph per LOSO fold via FCI,
using the causal-learn library, per your proposal's Section 4.4.2.

Install: pip install causal-learn

DATA-TYPE CAVEAT (read this before trusting the output):
CASME II's AU vectors (from au_features.py) are BINARY presence/absence,
not continuous activation values. Fisher's Z (the default test here, and
the one your proposal specifies to match Tan et al. [2025]) assumes
continuous, roughly-Gaussian data — technically violated by binary input.
In practice FCI+Fisher's Z is commonly run on binary data anyway in the
causal discovery literature (it still tends to pick up real dependence
structure), but it's a real limitation worth flagging in your thesis.
Two alternatives if you want to address it head-on rather than footnote
it:
  - Use `independence_test_method="chisq"` or "gsq" (both included in
    causal-learn, designed for discrete/binary data) instead of Fisher's Z.
  - Source continuous AU intensities from an AU-intensity detector (e.g.
    OpenFace's AU regression output) instead of the coding-sheet presence
    flags, if you want to keep Fisher's Z as specified in your proposal.
This module supports either — see `independence_test` argument.
"""

import numpy as np


def learn_fci_graph(au_matrix, vocab, alpha=0.05, independence_test="fisherz", verbose=False):
    """
    au_matrix: [N, len(vocab)] binary (or continuous) AU feature matrix
               for ONE LOSO fold's training subjects only — never include
               the held-out test subject's data here, exactly as with the
               CI-Block's prototype bank.
    vocab:     AU code list, same order as au_matrix's columns.
    alpha:     significance level for the independence test — your
               proposal treats this as a hyperparameter to sweep.
    independence_test: "fisherz" (continuous, matches Tan et al. [2025]
               exactly) or "chisq"/"gsq" (appropriate for CASME II's
               actual binary AU vectors — see module docstring).

    Returns: (A, pag_graph, edges)
        A: [N, N] binary symmetric adjacency — "an edge exists between
           these AU nodes" — this is what graph_ci_block.py consumes.
           NOTE: this collapses the PAG's richer edge-mark typology
           (circle/arrow/tail, representing (un)certainty about
           direction and latent confounding) down to plain presence/
           absence, because the CI-Block gating in this implementation
           doesn't yet use edge direction. Keep `pag_graph` around for
           your Section 4.7 structural analysis (precision/recall vs.
           hand-designed graph), which DOES care about the richer typing.
        pag_graph: the raw causal-learn Graph object (full PAG detail)
        edges: causal-learn's Edge list (human-readable, e.g. "AU1 o-o AU2")
    """
    from causallearn.search.ConstraintBased.FCI import fci

    if au_matrix.shape[0] < 20:
        # FCI needs enough samples per fold to estimate conditional
        # independencies reliably; CASME II's LOSO folds can be small.
        # This isn't a hard failure, just a warning worth surfacing.
        print(
            f"  WARNING: only {au_matrix.shape[0]} samples in this fold's "
            f"training set — FCI's independence tests may be unreliable "
            f"with this little data. This is directly relevant to your "
            f"proposal's Section 4.4.2 stability analysis."
        )

    pag_graph, edges = fci(
        au_matrix,
        independence_test_method=independence_test,
        alpha=alpha,
        node_names=[f"AU{a}" for a in vocab],
        show_progress=verbose,
        verbose=verbose,
    )

    A = (pag_graph.graph != 0).astype(float)
    np.fill_diagonal(A, 0)

    return A, pag_graph, edges


def graph_edge_overlap(A_learned, A_handdesigned):
    """
    Section 4.7: precision/recall of the learned graph against the
    hand-designed graph as reference. Both A_learned and A_handdesigned
    must already be aligned to the SAME vocab/order.
    """
    learned_edges = set()
    ref_edges = set()
    n = A_learned.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A_learned[i, j]:
                learned_edges.add((i, j))
            if A_handdesigned[i, j]:
                ref_edges.add((i, j))

    if not learned_edges:
        return {"precision": 0.0, "recall": 0.0, "n_learned": 0, "n_ref": len(ref_edges)}

    tp = len(learned_edges & ref_edges)
    precision = tp / len(learned_edges)
    recall = tp / len(ref_edges) if ref_edges else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "n_learned": len(learned_edges),
        "n_ref": len(ref_edges),
        "n_overlap": tp,
    }


def graph_stability_across_folds(adjacency_list):
    """
    Section 4.4.2/4.7: proportion of edges that appear consistently
    across all LOSO folds. adjacency_list: list of [N,N] arrays, all
    aligned to the same vocab/order.
    """
    stacked = np.stack(adjacency_list)  # [n_folds, N, N]
    n_folds = stacked.shape[0]
    always_present = (stacked.sum(axis=0) == n_folds).astype(float)
    ever_present = (stacked.sum(axis=0) > 0).astype(float)

    n = stacked.shape[1]
    iu = np.triu_indices(n, k=1)
    ever_count = ever_present[iu].sum()
    stable_count = always_present[iu].sum()

    stability = stable_count / ever_count if ever_count > 0 else 0.0
    return {
        "stability_ratio": stability,
        "n_ever_present": int(ever_count),
        "n_always_present": int(stable_count),
    }


if __name__ == "__main__":
    import argparse
    from scripts.au_features import build_au_matrix, discover_au_vocabulary
    from scripts.hand_designed_au_graph import build_adjacency

    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--test", choices=["fisherz", "chisq", "gsq"], default="chisq",
                         help="chisq/gsq are more appropriate for CASME II's binary "
                              "AU vectors; fisherz matches Tan et al. [2025] exactly "
                              "but assumes continuous data")
    args = parser.parse_args()

    vocab, _ = discover_au_vocabulary(args.xlsx)
    X, subjects, labels = build_au_matrix(args.xlsx, vocab)

    print(f"Learning FCI graph on full dataset (sanity check, not a real "
          f"LOSO fold) — {X.shape[0]} samples, {len(vocab)} AU nodes, "
          f"test={args.test}, alpha={args.alpha}")
    A, pag, edges = learn_fci_graph(X, vocab, alpha=args.alpha, independence_test=args.test)

    print(f"\nLearned {int(A.sum() / 2)} undirected edge(s):")
    for i in range(len(vocab)):
        for j in range(i + 1, len(vocab)):
            if A[i, j]:
                print(f"  AU{vocab[i]} -- AU{vocab[j]}")

    A_hand, _, _ = build_adjacency(vocab)
    overlap = graph_edge_overlap(A, A_hand)
    print(f"\nOverlap vs. hand-designed graph: {overlap}")
