"""
train_cimer_casme2.py — Wiring CIBlock (+ optional AU causal graph) into a
CASME II LOSO training loop
===========================================================================

This assumes:
  - Onset to apex and apex toffset optical flow
    images per sequence have been computed from compute_flow_casme2.py and saved as two image
    files per sequence, e.g.:
        <root>/sub01/EP02_01f/flow_onset_apex.png
        <root>/sub01/EP02_01f/flow_apex_offset.png
  - The CASME II coding spreadsheet is available
  - The Context-Cluster repo (ma-xu/Context-Cluster) is cloned and its
    `models` package is importable (so `coc_tiny` gets registered with
        timm).

Three experimental conditions:
  --graph_mode none          Plain CIBlock with no causal graph
  --graph_mode handdesigned  FACS-based AU graph (fixed across folds)
  --graph_mode fci           FCI-learned AU graph which is refit per fold from
                              that fold's training subjects only

Run: python train_cimer_casme2.py --data_root <flow_images_dir> --xlsx <coding_file.xlsx> --graph_mode handdesigned
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# Make sure Context-Cluster's `models` package is on the path and imported
# BEFORE calling timm.create_model, so coc_tiny is registered:
#   import sys; sys.path.append("/path/to/Context-Cluster")
import models  # noqa: F401  (registers coc_tiny with timm as a side effect)

from scripts.ci_block import CIBlock, ci_mer_loss
from scripts.graph_ci_block import GraphGatedCIBlock
from scripts.au_features import discover_au_vocabulary, build_au_matrix
from scripts.hand_designed_au_graph import build_adjacency as build_handdesigned_adjacency
from scripts.fci_graph_learning import learn_fci_graph, graph_edge_overlap, graph_stability_across_folds


CASME2_LABEL_TO_CLS = {
    "happiness": 0, "others": 1, "disgust": 2, "surprise": 3, "repression": 4,
}
NUM_CLASSES = len(CASME2_LABEL_TO_CLS)


# ---------------------------------------------------------------------------
# 1. Dataset — returns the two flow images AND the subject id, since the
#    CI-Block needs subject identity to build its prototypes. This is the
#    one addition CI-MER training needs beyond a normal classification loop.
# ---------------------------------------------------------------------------
class CASME2Flow(Dataset):
    def __init__(self, root, xlsx_path, held_out_subject, phase="train"):
        df = pd.read_excel(xlsx_path)
        df = df[df["Estimated Emotion"].isin(CASME2_LABEL_TO_CLS.keys())]
        if phase == "train":
            df = df[df["Subject"] != held_out_subject]
        else:
            df = df[df["Subject"] == held_out_subject]
        df = df.reset_index(drop=True)

        # compute_flow_casme2.py skips sequences it couldn't process (see
        # its flow_skip_log.csv)
        has_flow = []
        for _, row in df.iterrows():
            seq_dir = os.path.join(root, f"sub{int(row['Subject']):02d}", str(row["Filename"]))
            ok = (
                os.path.exists(os.path.join(seq_dir, "flow_onset_apex.png"))
                and os.path.exists(os.path.join(seq_dir, "flow_apex_offset.png"))
            )
            has_flow.append(ok)

        n_missing = len(df) - sum(has_flow)
        if n_missing > 0:
            print(f"  [{phase}] {n_missing}/{len(df)} sequences have no flow images on disk "
                  f"(see flow_skip_log.csv from compute_flow_casme2.py) — excluding them from this fold.")

        self.rows = df[pd.Series(has_flow)].reset_index(drop=True)
        self.root = root
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        seq_dir = os.path.join(
            self.root, f"sub{int(row['Subject']):02d}", str(row["Filename"])
        )
        flow1 = self.transform(
            Image.open(os.path.join(seq_dir, "flow_onset_apex.png")).convert("RGB")
        )
        flow2 = self.transform(
            Image.open(os.path.join(seq_dir, "flow_apex_offset.png")).convert("RGB")
        )
        label = CASME2_LABEL_TO_CLS[row["Estimated Emotion"]]
        subject = int(row["Subject"])
        return flow1, flow2, label, subject


# ---------------------------------------------------------------------------
# 2. Model wrapper — CoC-Tiny run twice (shared weights, once per flow
#    image), features concatenated. num_classes=0 makes CoC-Tiny return
#    its pooled 320-d feature vector instead of ImageNet logits.
# ---------------------------------------------------------------------------
class CIMERModel(nn.Module):
    def __init__(self, num_classes, pretrained_path=None, adjacency=None, au_dim=16):
        """
        adjacency: None -> plain CIBlock (condition 1, no causal graph).
                   [n_nodes, n_nodes] array -> GraphGatedCIBlock, gated by
                   this adjacency (condition 2 hand-designed, or condition
                   3 FCI-learned — same model class either way, only the
                   adjacency passed in differs).
        """
        super().__init__()
        self.backbone = timm.create_model("coc_tiny", pretrained=False, num_classes=0)
        self.pretrained_loaded_fraction = None
        if pretrained_path:
            # PyTorch >=2.6 defaults torch.load to weights_only=True, which
            # rejects checkpoints that bundle non-tensor metadata (this one
            # includes an argparse.Namespace of training args alongside the
            # weights).
            state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state_dict = state_dict.get("state_dict", state_dict.get("model", state_dict))

            own_keys = set(self.backbone.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            matched = own_keys & ckpt_keys

            result = self.backbone.load_state_dict(state_dict, strict=False)
            self.pretrained_loaded_fraction = len(matched) / max(len(own_keys), 1)

            print(f"Pretrained checkpoint: {len(matched)}/{len(own_keys)} backbone parameter "
                  f"tensors matched by name ({self.pretrained_loaded_fraction:.1%}).")
            if self.pretrained_loaded_fraction < 0.5:
                print("  WARNING: less than half the backbone matched — the checkpoint's layer "
                      "names likely don't match this version of the model code (a known issue "
                      "with Context-Cluster's repo history, see their GitHub issue #2). The "
                      "backbone is effectively training from random initialization, not from "
                      "ImageNet pretraining, even though loading didn't error out. This is worth "
                      "noting as a limitation if you keep going without resolving it, or you can "
                      "rerun with no --pretrained argument to make the random-init choice explicit "
                      "rather than incidental.")
            if result.missing_keys:
                print(f"  {len(result.missing_keys)} model keys had no match in the checkpoint "
                      f"(e.g. {result.missing_keys[:3]})")
            if result.unexpected_keys:
                print(f"  {len(result.unexpected_keys)} checkpoint keys had no match in the model "
                      f"(e.g. {result.unexpected_keys[:3]})")

        feature_dim = 320 * 2  # coc_tiny's last-stage embed dim, x2 for two flow images
        self.direct_classifier = nn.Linear(feature_dim, num_classes)

        if adjacency is None:
            self.ci_block = CIBlock(feature_dim=feature_dim, num_classes=num_classes)
        else:
            self.ci_block = GraphGatedCIBlock(
                backbone_feature_dim=feature_dim,
                num_classes=num_classes,
                adjacency=adjacency,
                au_dim=au_dim,
            )

    def extract_features(self, flow1, flow2):
        f1 = self.backbone(flow1)
        f2 = self.backbone(flow2)
        return torch.cat([f1, f2], dim=1)


# ---------------------------------------------------------------------------
# 3. Training loop for one LOSO fold
# ---------------------------------------------------------------------------
def train_one_fold(model, train_loader, val_loader, device, epochs=50, lr=1e-4, log_fn=None):
    """
    log_fn: optional callable(dict) -> None, called once per epoch with
            {"epoch": ..., "train_loss": ...}. Used for W&B logging without
            making this function depend on wandb directly — pass None to
            skip logging entirely.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for flow1, flow2, labels, subjects in train_loader:
            flow1, flow2, labels = flow1.to(device), flow2.to(device), labels.to(device)

            features = model.extract_features(flow1, flow2)
            direct_logits = model.direct_classifier(features)

            # Feed this batch into the running prototype bank (no grad).
            # Safe by construction: train_loader only ever contains the
            # training-fold subjects, so the held-out test subject never
            # contributes to its own debiasing prototypes.
            model.ci_block.update_prototypes(features, labels, subjects)

            ci_logits = model.ci_block(features, labels)
            loss = ci_mer_loss(direct_logits, ci_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Fold the batch-level accumulation into stable per-category
        # prototypes once per epoch (cheap; no need to do this every step).
        model.ci_block.refresh_category_prototypes()

        if log_fn is not None:
            log_fn({"epoch": epoch, "train_loss": sum(epoch_losses) / len(epoch_losses)})

    # ---- Evaluation: CI-Block is NOT used at test time (paper, Sec II-B:
    # "In the inference phase, the CI-Block can be removed") ----
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for flow1, flow2, labels, _subjects in val_loader:
            flow1, flow2 = flow1.to(device), flow2.to(device)
            features = model.extract_features(flow1, flow2)
            logits = model.direct_classifier(features)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


# ---------------------------------------------------------------------------
# 4. LOSO driver — a fresh model (and therefore a fresh, empty CIBlock
#    prototype bank) per fold, exactly like the original train_CASME2.py
#    pattern of reinstantiating the model per held-out subject.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pretrained", default=None, help="Path to CoC-Tiny ImageNet checkpoint")

    parser.add_argument("--graph_mode", choices=["none", "handdesigned", "fci"], default="none",
                         help="none=condition 1, handdesigned=condition 2, fci=condition 3 (proposal Sec 4.6)")
    parser.add_argument("--au_min_count", type=int, default=2,
                         help="drop AU nodes appearing fewer than this many times in the whole dataset")
    parser.add_argument("--au_dim", type=int, default=16, help="feature width per AU node in GraphGatedCIBlock")
    parser.add_argument("--fci_alpha", type=float, default=0.05)
    parser.add_argument("--fci_test", choices=["fisherz", "chisq", "gsq"], default="chisq",
                         help="chisq/gsq suit CASME II's binary AU vectors better than fisherz — see fci_graph_learning.py docstring")
    parser.add_argument("--save_graphs_dir", default=None,
                         help="if set, saves each fold's adjacency matrix here (needed for Sec 4.7 stability analysis)")
    parser.add_argument("--max_folds", type=int, default=None,
                         help="Limit to the first N LOSO folds — use this for smoke-testing "
                              "the pipeline (e.g. --max_folds 2 --epochs 2) before committing "
                              "to the full run across all subjects.")
    parser.add_argument("--results_csv", default="results.csv",
                         help="Every fold's result is appended here as it completes — a full "
                              "26-fold x 50-epoch run takes a long time on CPU, and console "
                              "output alone doesn't survive a closed terminal or let you "
                              "compare graph_mode conditions afterward.")
    parser.add_argument("--use_wandb", action="store_true",
                         help="Log per-epoch loss and per-fold acc/F1 to Weights & Biases. "
                              "Requires `pip install wandb` and being logged in (`wandb login`) "
                              "or setting WANDB_API_KEY. Off by default so the script has no "
                              "hard dependency on wandb or an internet connection.")
    parser.add_argument("--wandb_project", default="cimer-casme2",
                         help="W&B project name (only used if --use_wandb is set)")
    parser.add_argument("--wandb_run_name", default=None,
                         help="W&B run name (only used if --use_wandb is set); defaults to "
                              "'<graph_mode>' if not given")
    args = parser.parse_args()

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
        except ImportError:
            raise SystemExit("--use_wandb was set but wandb isn't installed. Run: pip install wandb")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.graph_mode,
            config=vars(args),
        )

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    device = torch.device("cpu")

    df = pd.read_excel(args.xlsx)
    subjects = sorted(df["Subject"].unique().tolist())
    if args.max_folds is not None:
        subjects = subjects[: args.max_folds]
        print(f"--max_folds set: running only {len(subjects)} fold(s): {subjects}")

    # --- AU vocabulary + hand-designed graph are built ONCE, globally,
    # not per fold. This is deliberate: both the hand-designed condition
    # and the FCI condition need to operate over the SAME set of AU nodes
    # across all folds, or fold results (and the Sec 4.7 precision/recall/
    # stability comparisons) aren't comparable to each other. ---
    au_vocab, handdesigned_adjacency = None, None
    if args.graph_mode in ("handdesigned", "fci"):
        au_vocab, _ = discover_au_vocabulary(args.xlsx, min_count=args.au_min_count)
        print(f"AU vocabulary ({len(au_vocab)} nodes): {au_vocab}")

    if args.graph_mode == "handdesigned":
        handdesigned_adjacency, used, dropped = build_handdesigned_adjacency(au_vocab)
        print(f"Hand-designed graph: {len(used)} usable edges, {len(dropped)} dropped (AU not in vocab)")

    if args.save_graphs_dir:
        os.makedirs(args.save_graphs_dir, exist_ok=True)

    fold_accs, fold_f1s = [], []
    fold_adjacencies = []  # for FCI stability-across-folds analysis (Sec 4.7)

    for fold_i, held_out in enumerate(subjects):
        print(f"*** fold {fold_i + 1}/{len(subjects)} — held-out subject {held_out} ***")

        train_ds = CASME2Flow(args.data_root, args.xlsx, held_out, phase="train")
        val_ds = CASME2Flow(args.data_root, args.xlsx, held_out, phase="test")

        if len(val_ds) == 0:
            print(f"  SKIPPING fold — held-out subject {held_out} has no sequences with "
                  f"flow images (all were skipped during flow computation). Can't evaluate "
                  f"a fold with zero validation samples.")
            continue
        if len(train_ds) < args.batch_size:
            print(f"  SKIPPING fold — only {len(train_ds)} training samples available, "
                  f"fewer than --batch_size {args.batch_size} (drop_last=True would yield "
                  f"zero batches). Lower --batch_size or check flow_skip_log.csv for this fold's subjects.")
            continue

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=4)

        # --- Pick this fold's adjacency ---
        adjacency = None
        if args.graph_mode == "handdesigned":
            adjacency = handdesigned_adjacency

        elif args.graph_mode == "fci":
            # Refit FCI on THIS fold's training subjects only — never the
            # held-out subject, mirroring the CI-Block's own leakage
            # safeguard, and directly enabling the proposal's per-fold
            # stability analysis (Sec 4.4.2 / 4.7).
            fold_df_mask = df["Subject"] != held_out
            X_train_au, _, _ = build_au_matrix(args.xlsx, au_vocab)
            X_train_au = X_train_au[fold_df_mask.to_numpy()]

            adjacency, pag, edges = learn_fci_graph(
                X_train_au, au_vocab, alpha=args.fci_alpha, independence_test=args.fci_test,
            )
            fold_adjacencies.append(adjacency)
            print(f"  FCI learned {int(adjacency.sum() / 2)} edges on {X_train_au.shape[0]} training samples")

        if args.save_graphs_dir and adjacency is not None:
            np.save(os.path.join(args.save_graphs_dir, f"fold_{held_out}_adjacency.npy"), adjacency)

        model = CIMERModel(
            NUM_CLASSES, pretrained_path=args.pretrained, adjacency=adjacency, au_dim=args.au_dim
        ).to(device)

        def epoch_log_fn(metrics, _fold=fold_i, _held_out=held_out):
            if wandb_run is not None:
                wandb_run.log({"fold": _fold, "held_out_subject": _held_out, **metrics})

        acc, f1 = train_one_fold(
            model, train_loader, val_loader, device, epochs=args.epochs,
            log_fn=epoch_log_fn if wandb_run is not None else None,
        )
        print(f"  -> acc={acc:.4f}, macro-F1={f1:.4f}")
        fold_accs.append(acc)
        fold_f1s.append(f1)

        if wandb_run is not None:
            wandb_run.log({
                "fold": fold_i, "held_out_subject": held_out,
                "fold_acc": acc, "fold_macro_f1": f1,
            })

        # Write immediately, not just at the end — a 26-fold x 50-epoch CPU
        # run can take hours; don't lose completed folds if it's interrupted.
        result_row = pd.DataFrame([{
            "graph_mode": args.graph_mode,
            "fold": fold_i + 1,
            "held_out_subject": held_out,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "acc": acc,
            "macro_f1": f1,
            "pretrained_match_fraction": model.pretrained_loaded_fraction,
        }])
        write_header = not os.path.exists(args.results_csv)
        result_row.to_csv(args.results_csv, mode="a", header=write_header, index=False)

    print(f"\nLOSO mean acc: {np.mean(fold_accs):.4f}, mean macro-F1: {np.mean(fold_f1s):.4f}")
    print(f"Per-fold results written to: {args.results_csv}")

    if wandb_run is not None:
        wandb_run.log({
            "loso_mean_acc": np.mean(fold_accs),
            "loso_mean_macro_f1": np.mean(fold_f1s),
        })

    # --- Sec 4.7 structural analysis, only meaningful for the FCI condition ---
    if args.graph_mode == "fci" and len(fold_adjacencies) > 1:
        stability = graph_stability_across_folds(fold_adjacencies)
        print(f"\nFCI graph stability across folds: {stability}")

        if handdesigned_adjacency is None:
            # Build it now purely as a reference for the overlap comparison,
            # even though this run's actual training used FCI graphs.
            handdesigned_adjacency, _, _ = build_handdesigned_adjacency(au_vocab)
        mean_overlap_precision, mean_overlap_recall = [], []
        for A in fold_adjacencies:
            ov = graph_edge_overlap(A, handdesigned_adjacency)
            mean_overlap_precision.append(ov["precision"])
            mean_overlap_recall.append(ov["recall"])
        print(f"FCI vs hand-designed graph — mean precision: {np.mean(mean_overlap_precision):.3f}, "
              f"mean recall: {np.mean(mean_overlap_recall):.3f}")

        if wandb_run is not None:
            wandb_run.log({
                "fci_stability_ratio": stability["stability_ratio"],
                "fci_vs_handdesigned_precision": np.mean(mean_overlap_precision),
                "fci_vs_handdesigned_recall": np.mean(mean_overlap_recall),
            })

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()