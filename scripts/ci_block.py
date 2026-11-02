"""
CI-Block — Causal Intervention module for Micro-Expression Recognition
========================================================================

A from-scratch PyTorch reimplementation of the CI-Block described in:

    Fang, X., Wu, Q., & Cao, L. (2025). CI-MER: A Novel Causal Intervention
    Framework For Micro-Expression Recognition. ICME 2025.

There is no public code for this paper (verified — no repo, no code-
availability statement). This implementation follows Eqs. (1)-(5) and the
description in Section II of the paper directly:

  - Causal graph: S (subject-specific features) -> X (universal motion
    features) -> Y (emotion label), plus a direct S -> Y confound path the
    paper wants removed via backdoor adjustment.
  - Eq. (1): P(Y|do(X)) = sum_i P(Y | X=x, S=s_i) * P(s_i)
  - Eq. (2)-(3): NWGM approximation — instead of summing over every
    subject explicitly, precompute one "subject-habit prototype" feature
    per (subject, category) pair, weighted by that subject's share of
    samples in that category, then average these into one prototype per
    category: fea_cls.
  - Eq. (4): fuse the sample's own feature with its category's prototype
    and classify: P(Y|do(X)) = W(fea_x + fea_cls)
  - Eq. (5): total loss = CE(P(Y|do(X)), y) + CE(P(Y|X), y) — the CI-Block
    branch is an auxiliary/training-only signal; a separate direct
    classifier head (not shown here — just a plain nn.Linear on fea_x)
    handles the actual X -> Y estimate. At test time, drop the CI-Block.

This is a plug-in module: it consumes whatever per-sample feature vector
your backbone produces (CoC-Tiny, ResNet, CoDER's transformer+LSTM output,
etc.) — it doesn't care how that feature was made.
"""

import torch
import torch.nn as nn


class CIBlock(nn.Module):
    """
    Causal intervention module (backdoor adjustment via NWGM approximation).

    Usage per training step:
        1. Run the backbone forward for the whole batch -> features [B, D]
        2. Call `update_prototypes(features.detach(), labels, subjects)`
           to accumulate this batch into the running per-(subject,category)
           prototype bank (no-grad, cheap).
        3. Periodically (e.g. once per epoch, or every N steps) call
           `refresh_category_prototypes()` to fold the bank into the
           per-category prototypes used at training time.
        4. Call `forward(features, labels)` to get the do(X) logits for
           the auxiliary CI loss term.
    At inference: just use your normal direct classifier on `features`;
    don't call this module at all.
    """

    def __init__(self, feature_dim, num_classes, momentum=0.9):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.momentum = momentum

        # W in Eq. (4) — classifier applied to (fea_x + fea_cls)
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Running per-category prototype bank: one D-dim vector per class.
        # Not a learnable parameter — updated via EMA from data, per the
        # paper's description of inferring these "without preserving the
        # gradient" (Section II-B).
        self.register_buffer(
            "category_prototypes", torch.zeros(num_classes, feature_dim)
        )
        self.register_buffer(
            "initialized", torch.zeros(num_classes, dtype=torch.bool)
        )

        # Per-(subject, category) running sums, used to build the
        # subject-weighted prototype before folding into category_prototypes.
        # Keyed lazily since the number of subjects varies by dataset/fold.
        self._subject_cls_sum = {}   # (subject_id, cls) -> running mean tensor
        self._subject_cls_count = {}  # (subject_id, cls) -> sample count

    @torch.no_grad()
    def update_prototypes(self, features, labels, subjects):
        """
        Accumulate a batch of (already backbone-extracted) features into
        the per-(subject, category) running means.

        features: [B, D] tensor (detached, no grad needed — Section II-B:
                  "deep features can be inferred through the backbone
                  directly without preserving the gradient")
        labels:   [B] long tensor of class indices
        subjects: [B] list/tensor of subject identifiers (any hashable)
        """
        features = features.detach()
        for i in range(features.size(0)):
            key = (int(subjects[i]), int(labels[i]))
            feat = features[i]
            if key not in self._subject_cls_sum:
                self._subject_cls_sum[key] = feat.clone()
                self._subject_cls_count[key] = 1
            else:
                n = self._subject_cls_count[key]
                self._subject_cls_sum[key] = (
                    self._subject_cls_sum[key] * n + feat
                ) / (n + 1)
                self._subject_cls_count[key] = n + 1

    @torch.no_grad()
    def refresh_category_prototypes(self):
        """
        Fold the per-(subject, category) means into one prototype per
        category, weighted by each subject's share of samples within that
        category (Eq. 3: fea_cls = sum_i [ (1/n) * sum_n s_i_cls_n * P(s_i_cls) ]).

        Call this periodically (e.g. once per epoch) rather than every
        step — the paper doesn't require it to be perfectly fresh, and
        recomputing every step is wasted compute.
        """
        # Group by category
        totals_per_cls = {}
        for (subject_id, cls), count in self._subject_cls_count.items():
            totals_per_cls.setdefault(cls, 0)
            totals_per_cls[cls] += count

        new_prototypes = torch.zeros_like(self.category_prototypes)
        seen = torch.zeros(self.num_classes, dtype=torch.bool)

        for (subject_id, cls), count in self._subject_cls_count.items():
            weight = count / totals_per_cls[cls]  # P(s_i | cls)
            new_prototypes[cls] += weight * self._subject_cls_sum[(subject_id, cls)]
            seen[cls] = True

        # EMA update for stability across epochs/folds, only for classes
        # we actually saw data for this round
        for cls in range(self.num_classes):
            if not seen[cls]:
                continue
            if not self.initialized[cls]:
                self.category_prototypes[cls] = new_prototypes[cls]
                self.initialized[cls] = True
            else:
                self.category_prototypes[cls] = (
                    self.momentum * self.category_prototypes[cls]
                    + (1 - self.momentum) * new_prototypes[cls]
                )

    def forward(self, features, labels):
        """
        Eq. (4): P(Y|do(X)) = W(fea_x + fea_cls)

        features: [B, D] — this batch's own (gradient-carrying) features
        labels:   [B] — used to look up the correct category prototype
                  per sample
        returns:  [B, num_classes] logits for the auxiliary CI loss
        """
        proto = self.category_prototypes[labels]  # [B, D], no grad through this
        fused = features + proto
        return self.classifier(fused)


def ci_mer_loss(direct_logits, ci_logits, labels, ce_loss_fn=None):
    """
    Eq. (5): Loss = CE(P(Y|do(X))) + CE(P(Y|X))

    direct_logits: output of your model's normal classifier head on `features`
    ci_logits:     output of CIBlock.forward(features, labels)
    """
    if ce_loss_fn is None:
        ce_loss_fn = nn.CrossEntropyLoss()
    return ce_loss_fn(direct_logits, labels) + ce_loss_fn(ci_logits, labels)


# ---------------------------------------------------------------------------
# Minimal worked example — wiring CIBlock onto an arbitrary backbone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, D, num_classes = 8, 256, 5

    # Stand-in for whatever backbone you use (CoC-Tiny, CoDER's
    # transformer+LSTM, etc.) — just needs to emit a [B, D] feature vector.
    backbone = nn.Sequential(nn.Linear(64, D), nn.ReLU())
    direct_classifier = nn.Linear(D, num_classes)
    ci_block = CIBlock(feature_dim=D, num_classes=num_classes)

    optimizer = torch.optim.Adam(
        list(backbone.parameters())
        + list(direct_classifier.parameters())
        + list(ci_block.parameters()),
        lr=1e-4,
    )

    dummy_input = torch.randn(B, 64)
    labels = torch.randint(0, num_classes, (B,))
    subjects = torch.randint(0, 10, (B,))  # 10 fake subject IDs

    for epoch in range(3):
        features = backbone(dummy_input)

        direct_logits = direct_classifier(features)

        # Update the running prototype bank with this batch (no grad)
        ci_block.update_prototypes(features, labels, subjects)
        ci_block.refresh_category_prototypes()

        ci_logits = ci_block(features, labels)

        loss = ci_mer_loss(direct_logits, ci_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch {epoch}: loss={loss.item():.4f}")

    print("\nAt inference, drop CIBlock entirely:")
    with torch.no_grad():
        test_features = backbone(dummy_input)
        test_logits = direct_classifier(test_features)
        print(test_logits.argmax(dim=1))
