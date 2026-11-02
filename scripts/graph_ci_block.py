"""
graph_ci_block.py — Graph-gated CI-Block: the hand-designed-graph and
FCI-learned-graph experimental conditions (proposal Section 4.5, 4.6
conditions 2 & 3). Condition 1 ("no causal graph") is plain CIBlock from
ci_block.py, unchanged.

DESIGN NOTE — this is my architectural proposal, not something your
source papers specify exactly: neither CI-MER nor your proposal gives a
literal formula for "edges constraining which feature interactions are
treated as causal." What follows is one concrete, defensible way to
realize that sentence:

  1. Project the backbone's opaque feature vector into an AU-aligned
     space: one `au_dim`-sized chunk per AU node (`au_projection`).
  2. Run a single GCN-style propagation step using the (hand-designed or
     FCI-learned) adjacency matrix: each AU node's feature becomes a
     normalized aggregate of its graph-neighbors' features. This is
     exactly "edges constraining which feature interactions are treated
     as causal" — an AU with no edge to another AU never mixes features
     with it; the baseline (ci_block.py, condition 1) is equivalent to
     this with a fully-disconnected graph (identity propagation).
  3. Flatten back to a vector and feed it into the SAME prototype-
     averaging / fusion / classification logic as CIBlock (Eqs. 3-4) —
     inherited via subclassing, not reimplemented.

If your supervisor or the literature review turns up a more specific
mechanism, this is the piece to swap — everything else (AU extraction,
graph construction, training loop) stays the same regardless of exactly
how the gating math works.
"""

import numpy as np
import torch
import torch.nn as nn

from scripts.ci_block import CIBlock


def normalize_adjacency(A):
    """
    Symmetric normalization with self-loops, standard GCN preprocessing
    (Kipf & Welling, 2017): A_norm = D^-1/2 (A + I) D^-1/2
    """
    A = A.copy().astype(np.float32)
    n = A.shape[0]
    A_hat = A + np.eye(n, dtype=np.float32)
    D = A_hat.sum(axis=1)
    D_inv_sqrt = np.zeros_like(D)
    nonzero = D > 0
    D_inv_sqrt[nonzero] = np.power(D[nonzero], -0.5)
    D_mat = np.diag(D_inv_sqrt)
    return D_mat @ A_hat @ D_mat


class GraphGatedCIBlock(CIBlock):
    def __init__(self, backbone_feature_dim, num_classes, adjacency, au_dim=16, momentum=0.9):
        """
        backbone_feature_dim: dim of the raw backbone feature (e.g. 640
                               for CoC-Tiny's two-flow-image concat output)
        adjacency: [n_nodes, n_nodes] numpy array — from
                   hand_designed_au_graph.build_adjacency() or
                   fci_graph_learning.learn_fci_graph()
        au_dim:    feature width allocated per AU node. Total graph-space
                   feature dim = n_nodes * au_dim, which becomes the
                   `feature_dim` the inherited CIBlock machinery operates on.
        """
        n_nodes = adjacency.shape[0]
        graph_feature_dim = n_nodes * au_dim
        super().__init__(feature_dim=graph_feature_dim, num_classes=num_classes, momentum=momentum)

        self.n_nodes = n_nodes
        self.au_dim = au_dim
        self.au_projection = nn.Linear(backbone_feature_dim, graph_feature_dim)
        self.gcn_weight = nn.Linear(au_dim, au_dim)

        adj_norm = normalize_adjacency(adjacency)
        self.register_buffer("adj_norm", torch.from_numpy(adj_norm))

    def graph_features(self, backbone_features):
        """
        backbone_features: [B, backbone_feature_dim]
        returns: [B, n_nodes * au_dim] graph-contextualized features
        """
        x = self.au_projection(backbone_features)          # [B, n_nodes*au_dim]
        x = x.view(-1, self.n_nodes, self.au_dim)           # [B, n_nodes, au_dim]
        # Propagate along graph edges: each node's feature becomes a
        # normalized aggregate of itself + its neighbors, per adjacency.
        x = torch.einsum("nm,bmd->bnd", self.adj_norm, x)   # [B, n_nodes, au_dim]
        x = torch.relu(self.gcn_weight(x))
        return x.reshape(x.size(0), -1)                     # [B, n_nodes*au_dim]

    def update_prototypes(self, backbone_features, labels, subjects):
        graph_feat = self.graph_features(backbone_features)
        super().update_prototypes(graph_feat, labels, subjects)

    def forward(self, backbone_features, labels):
        graph_feat = self.graph_features(backbone_features)
        return super().forward(graph_feat, labels)


if __name__ == "__main__":
    # Smoke test: random adjacency standing in for hand-designed/FCI graphs
    torch.manual_seed(0)
    B, backbone_dim, num_classes, n_nodes = 8, 640, 5, 13

    A = (np.random.rand(n_nodes, n_nodes) < 0.15).astype(float)
    A = np.triu(A, 1)
    A = A + A.T  # symmetric

    block = GraphGatedCIBlock(backbone_feature_dim=backbone_dim, num_classes=num_classes, adjacency=A, au_dim=16)
    direct_classifier = nn.Linear(backbone_dim, num_classes)

    optimizer = torch.optim.Adam(
        list(block.parameters()) + list(direct_classifier.parameters()), lr=1e-4
    )

    from scripts.ci_block import ci_mer_loss

    for epoch in range(3):
        features = torch.randn(B, backbone_dim)
        labels = torch.randint(0, num_classes, (B,))
        subjects = torch.randint(0, 10, (B,))

        direct_logits = direct_classifier(features)
        block.update_prototypes(features, labels, subjects)
        block.refresh_category_prototypes()
        ci_logits = block(features, labels)

        loss = ci_mer_loss(direct_logits, ci_logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch}: loss={loss.item():.4f}")

    print("\nGraphGatedCIBlock smoke test passed.")
