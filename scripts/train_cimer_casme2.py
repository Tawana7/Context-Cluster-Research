"""
train_cimer_casme2.py — Wiring CIBlock into a CASME II LOSO training loop
===========================================================================

This assumes:
  - You've already computed onset->apex and apex->offset optical flow
    images per sequence (see the CoC-Tiny/FlowNet2.0 discussion — TV-L1
    via OpenCV is the CPU-friendly choice), saved as two image files per
    sequence, e.g.:
        <root>/sub01/EP02_01f/flow_onset_apex.png
        <root>/sub01/EP02_01f/flow_apex_offset.png
  - You have the CASME II coding spreadsheet (Subject, Filename, Estimated
    Emotion, etc. — same columns analyze_casme2.py reads).
  - Context-Cluster repo (ma-xu/Context-Cluster) is cloned and its
    `models` package is importable (so `coc_tiny` gets registered with
    timm).

Run: python train_cimer_casme2.py --data_root <flow_images_dir> --xlsx <coding_file.xlsx>
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
        self.rows = df.reset_index(drop=True)
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
    def __init__(self, num_classes, pretrained_path=None):
        super().__init__()
        self.backbone = timm.create_model("coc_tiny", pretrained=False, num_classes=0)
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            state_dict = state_dict.get("state_dict", state_dict.get("model", state_dict))
            self.backbone.load_state_dict(state_dict, strict=False)

        feature_dim = 320 * 2  # coc_tiny's last-stage embed dim, x2 for two flow images
        self.direct_classifier = nn.Linear(feature_dim, num_classes)
        self.ci_block = CIBlock(feature_dim=feature_dim, num_classes=num_classes)

    def extract_features(self, flow1, flow2):
        f1 = self.backbone(flow1)
        f2 = self.backbone(flow2)
        return torch.cat([f1, f2], dim=1)


# ---------------------------------------------------------------------------
# 3. Training loop for one LOSO fold
# ---------------------------------------------------------------------------
def train_one_fold(model, train_loader, val_loader, device, epochs=50, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
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

        # Fold the batch-level accumulation into stable per-category
        # prototypes once per epoch (cheap; no need to do this every step).
        model.ci_block.refresh_category_prototypes()

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
    args = parser.parse_args()

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    device = torch.device("cpu")

    df = pd.read_excel(args.xlsx)
    subjects = sorted(df["Subject"].unique().tolist())

    fold_accs, fold_f1s = [], []
    for fold_i, held_out in enumerate(subjects):
        print(f"*** fold {fold_i + 1}/{len(subjects)} — held-out subject {held_out} ***")

        train_ds = CASME2Flow(args.data_root, args.xlsx, held_out, phase="train")
        val_ds = CASME2Flow(args.data_root, args.xlsx, held_out, phase="test")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=4)

        model = CIMERModel(NUM_CLASSES, pretrained_path=args.pretrained).to(device)

        acc, f1 = train_one_fold(model, train_loader, val_loader, device, epochs=args.epochs)
        print(f"  -> acc={acc:.4f}, macro-F1={f1:.4f}")
        fold_accs.append(acc)
        fold_f1s.append(f1)

    print(f"\nLOSO mean acc: {np.mean(fold_accs):.4f}, mean macro-F1: {np.mean(fold_f1s):.4f}")


if __name__ == "__main__":
    main()
