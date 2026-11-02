"""
hand_designed_au_graph.py — FACS-anatomy-based AU causal graph.

IMPORTANT: your proposal cites Tan et al. [2026] as the specific source
for the hand-designed FACS-based AU graph model you intend to use. I don't
have access to that paper, so the edge list below is a *reasonable
starting point* built from well-established, widely-documented anatomical
relationships in the FACS literature (shared muscle groups, well-known
co-activation patterns) — not a reproduction of Tan et al.'s specific
graph. Treat this as scaffolding: swap `HAND_DESIGNED_EDGES` for the
actual edges from Tan et al. [2026] once you have that paper's graph, and
everything downstream (adjacency construction, the graph-gated CI-Block)
will keep working unchanged since it only depends on the edge list format.

Each edge is (AU_a, AU_b, rationale). Undirected by construction here —
the graph-gated CI-Block (see graph_ci_block.py) uses these as symmetric
"these AUs' features are allowed to interact" constraints, since directing
every edge correctly requires the specific causal-priority claims a
dedicated FACS-graph paper like Tan et al. [2026] would provide, and
getting that direction wrong is worse than leaving it undirected.
"""

HAND_DESIGNED_EDGES = [
    ("1", "2", "Inner and outer brow raiser — both driven by frontalis, "
                "almost always co-activate"),
    ("1", "4", "Inner brow raiser vs brow lowerer — antagonistic pair, "
               "frequently co-coded together in brow-furrowing expressions"),
    ("4", "7", "Brow lowerer and lid tightener — corrugator/orbicularis "
               "oculi share the fear/anger brow-and-eye complex"),
    ("6", "7", "Cheek raiser and lid tightener — both orbicularis oculi, "
               "share the same muscle"),
    ("6", "12", "Cheek raiser and lip corner puller — the Duchenne smile "
                "marker, one of the most well-documented AU co-activations"),
    ("9", "10", "Nose wrinkler and upper lip raiser — both driven by "
                "levator labii superioris alaeque nasi, classic disgust pair"),
    ("9", "4", "Nose wrinkler often co-occurs with brow lowering in "
               "disgust expressions"),
    ("15", "17", "Lip corner depressor and chin raiser — common "
                 "sadness/disgust pairing via mentalis and depressor anguli oris"),
    ("17", "24", "Chin raiser and lip pressor — mentalis interacts with "
                 "orbicularis oris in lip-tightening expressions"),
    ("23", "24", "Lip tightener and lip pressor — both orbicularis oris "
                 "actions, frequently coded together"),
    ("20", "25", "Lip stretcher and lips part — risorius action commonly "
                 "co-occurs with mouth opening in fear/surprise"),
    ("25", "26", "Lips part and jaw drop — sequential/graded actions of "
                 "mouth opening, near-always co-occur"),
    ("1", "5", "Inner brow raiser and upper lid raiser — frontalis and "
               "levator palpebrae share the surprise/fear eye-widening complex"),
    ("2", "5", "Outer brow raiser and upper lid raiser — same surprise "
               "complex as above"),
]


def get_au_vocab_from_edges():
    aus = set()
    for a, b, _ in HAND_DESIGNED_EDGES:
        aus.add(a)
        aus.add(b)
    return sorted(aus, key=int)


def build_adjacency(vocab):
    """
    vocab: the AU vocabulary discovered from your actual dataset
           (au_features.discover_au_vocabulary), NOT get_au_vocab_from_edges()
           — the two won't be identical, since your dataset may not contain
           every AU this hand-designed graph knows about, and vice versa.

    Returns an [N, N] symmetric 0/1 numpy array aligned to `vocab`'s order.
    Edges between AUs not present in `vocab` are silently dropped.
    """
    import numpy as np
    n = len(vocab)
    idx = {au: i for i, au in enumerate(vocab)}
    A = np.zeros((n, n))
    used_edges, dropped_edges = [], []

    for a, b, rationale in HAND_DESIGNED_EDGES:
        if a in idx and b in idx:
            A[idx[a], idx[b]] = 1
            A[idx[b], idx[a]] = 1
            used_edges.append((a, b))
        else:
            dropped_edges.append((a, b))

    return A, used_edges, dropped_edges


if __name__ == "__main__":
    import argparse
    from scripts.au_features import discover_au_vocabulary

    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--min_count", type=int, default=2)
    args = parser.parse_args()

    vocab, _ = discover_au_vocabulary(args.xlsx, args.min_count)
    A, used, dropped = build_adjacency(vocab)

    print(f"Dataset AU vocab ({len(vocab)}): {vocab}")
    print(f"\nHand-designed edges usable with this dataset ({len(used)}):")
    for a, b in used:
        print(f"  AU{a} -- AU{b}")
    if dropped:
        print(f"\nHand-designed edges dropped, AU not in dataset vocab ({len(dropped)}):")
        for a, b in dropped:
            print(f"  AU{a} -- AU{b}")
    print(f"\nAdjacency matrix shape: {A.shape}, density: {A.mean():.3f}")
