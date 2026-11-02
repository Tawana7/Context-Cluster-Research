r"""
compute_flow_casme2.py — CPU-friendly optical flow, no CUDA required
======================================================================

Replaces FlowNet2.0 for generating the two flow images per CASME II
sequence (onset->apex, apex->offset) that train_cimer_casme2.py expects.

Three methods, pick with --method:
  - "tvl1"       (default, recommended) — cv2.optflow.DualTVL1OpticalFlow.
                 This is the exact classical baseline the CI-MER paper
                 itself compares FlowNet2.0 against; edge-preserving,
                 good quality, pure CPU. Needs opencv-contrib-python.
  - "farneback"  — cv2.calcOpticalFlowFarneback. Built into plain
                 opencv-python (no -contrib needed), faster, slightly
                 blurrier flow fields. Good fallback if you hit an
                 opencv-contrib-python install issue.
  - "raft"       — torchvision's RAFT (raft_small). Modern learned flow,
                 runs on CPU but is noticeably slower per pair than the
                 two classical methods above. Only worth it if you
                 specifically want to compare flow quality against
                 TV-L1/Farneback in your ablations.

Install:
    pip install opencv-contrib-python pandas openpyxl numpy
    # only if you also want --method raft:
    pip install torch torchvision

Usage:
    python compute_flow_casme2.py \
        --xlsx "C:\path\to\CASME2-coding-20140508.xlsx" \
        --frames_root "C:\path\to\CASME2\Cropped" \
        --out_root "C:\path\to\CASME2\Flow" \
        --method tvl1

Frame naming assumption: <frames_root>/subNN/<Filename>/reg_img<K>.jpg
(matches the CoDER repo's convention — dataset.py builds this same path).
If your raw CASME II release uses a different naming pattern, adjust
`frame_path()` below.
"""

import argparse
import os
import re

import cv2
import numpy as np
import pandas as pd


def imread_unicode(path):
    """
    cv2.imread() silently fails (returns None) on Windows when the path
    contains non-ASCII characters — e.g. "CASME Ⅱ" uses the Unicode Roman
    numeral character U+2161, which OpenCV's underlying Windows file-open
    call can't represent in the default codepage. This reads the file via
    Python's own (Unicode-safe) file I/O instead, then decodes the bytes
    with OpenCV. Works identically to cv2.imread on Linux/Mac too.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except FileNotFoundError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """Unicode-safe counterpart to cv2.imwrite, same reasoning as above."""
    ext = os.path.splitext(path)[1]
    success, encoded = cv2.imencode(ext, img)
    if not success:
        raise IOError(f"Failed to encode image for {path}")
    encoded.tofile(path)


def frame_path(frames_root, subject, filename, frame_idx):
    return os.path.join(
        frames_root, f"sub{int(subject):02d}", str(filename), f"reg_img{int(frame_idx)}.jpg"
    )


def flow_to_rgb(flow):
    """
    Standard optical-flow-to-color encoding (HSV color wheel): direction
    of motion -> hue, magnitude -> brightness. This is the same encoding
    used across the MER literature (MDMO, Off-ApexNet, etc.) to turn a
    2-channel (dx, dy) flow field into a 3-channel image a CNN/backbone
    can consume.
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_flow(img1, img2, method):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if method == "tvl1":
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = tvl1.calc(gray1, gray2, None)
    elif method == "farneback":
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
    elif method == "raft":
        flow = compute_flow_raft(img1, img2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return flow_to_rgb(flow)


# Lazily initialized so torch/torchvision are only required if --method raft
_raft_model = None


def compute_flow_raft(img1, img2):
    global _raft_model
    import torch
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    from torchvision.transforms.functional import to_tensor

    if _raft_model is None:
        weights = Raft_Small_Weights.DEFAULT
        _raft_model = raft_small(weights=weights).eval()

    t1 = to_tensor(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    t2 = to_tensor(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    # RAFT expects sizes divisible by 8; CASME II crops are already square
    # (e.g. 224x224 or 256x256) so this is usually a no-op resize.
    h, w = t1.shape[-2:]
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    t1 = torch.nn.functional.interpolate(t1, size=(h8, w8))
    t2 = torch.nn.functional.interpolate(t2, size=(h8, w8))

    with torch.no_grad():
        flow_list = _raft_model(t1, t2)
    flow = flow_list[-1][0].permute(1, 2, 0).numpy()
    return flow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--frames_root", required=True)
def find_max_available_frame(frames_root, subject, filename):
    """Scan the sequence folder and return the highest reg_img<N>.jpg index
    actually present on disk, or None if the folder doesn't exist / is empty."""
    seq_dir = os.path.join(frames_root, f"sub{int(subject):02d}", str(filename))
    if not os.path.isdir(seq_dir):
        return None
    max_idx = None
    for f in os.listdir(seq_dir):
        m = re.match(r"reg_img(\d+)\.jpg$", f, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            max_idx = idx if max_idx is None else max(max_idx, idx)
    return max_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--method", choices=["tvl1", "farneback", "raft"], default="tvl1")
    parser.add_argument("--clamp_to_last_frame", action="store_true",
                         help="If OnsetFrame/ApexFrame/OffsetFrame exceeds the highest "
                              "reg_img<N>.jpg actually present for that sequence (a known "
                              "CASME II coding-sheet quirk), use the last available frame "
                              "instead of skipping the sequence entirely. Off by default — "
                              "skipping is the more conservative choice; clamping trades a "
                              "few slightly-inaccurate flow images for a few more retained "
                              "sequences, and which is preferable is a methodological call "
                              "you should make deliberately, not one the script should make "
                              "silently.")
    parser.add_argument("--log_path", default="flow_skip_log.csv",
                         help="Every skip or clamp gets written here, so you get a clean "
                              "summary at the end instead of scrolling console output.")
    args = parser.parse_args()

    df = pd.read_excel(args.xlsx)

    n_ok, n_skipped, n_clamped = 0, 0, 0
    log_rows = []

    for _, row in df.iterrows():
        subject, filename = row["Subject"], row["Filename"]
        onset, apex, offset = row["OnsetFrame"], row["ApexFrame"], row["OffsetFrame"]

        if apex == "/" or pd.isna(apex):
            apex = int((onset + offset) / 2)

        out_dir = os.path.join(args.out_root, f"sub{int(subject):02d}", str(filename))
        os.makedirs(out_dir, exist_ok=True)

        # Check requested frame indices against what's actually on disk BEFORE
        # trying to load anything, so we can log a clear reason either way.
        max_available = find_max_available_frame(args.frames_root, subject, filename)
        requested_frames = {"onset": int(onset), "apex": int(apex), "offset": int(offset)}
        out_of_range = {k: v for k, v in requested_frames.items()
                         if max_available is not None and v > max_available}

        if out_of_range:
            if args.clamp_to_last_frame and max_available is not None:
                for k in out_of_range:
                    requested_frames[k] = max_available
                log_rows.append({
                    "subject": subject, "filename": filename, "action": "clamped",
                    "detail": f"clamped {list(out_of_range.keys())} to frame {max_available}",
                })
                n_clamped += 1
            else:
                log_rows.append({
                    "subject": subject, "filename": filename, "action": "skipped",
                    "detail": f"requested {out_of_range}, but only {max_available} frames "
                              f"available — rerun with --clamp_to_last_frame to keep this "
                              f"sequence instead of skipping it",
                })
                n_skipped += 1
                continue

        onset, apex, offset = requested_frames["onset"], requested_frames["apex"], requested_frames["offset"]

        try:
            img_onset = imread_unicode(frame_path(args.frames_root, subject, filename, onset))
            img_apex = imread_unicode(frame_path(args.frames_root, subject, filename, apex))
            img_offset = imread_unicode(frame_path(args.frames_root, subject, filename, offset))

            if img_onset is None or img_apex is None or img_offset is None:
                raise FileNotFoundError("one or more frames missing")

            flow_oa = compute_flow(img_onset, img_apex, args.method)
            flow_ao = compute_flow(img_apex, img_offset, args.method)

            imwrite_unicode(os.path.join(out_dir, "flow_onset_apex.png"), flow_oa)
            imwrite_unicode(os.path.join(out_dir, "flow_apex_offset.png"), flow_ao)
            n_ok += 1
        except Exception as e:
            log_rows.append({
                "subject": subject, "filename": filename, "action": "skipped",
                "detail": str(e),
            })
            n_skipped += 1

    pd.DataFrame(log_rows).to_csv(args.log_path, index=False)

    print(f"\nDone. {n_ok} sequences processed, {n_clamped} clamped, {n_skipped} skipped.")
    print(f"Flow images written under: {args.out_root}")
    print(f"Full skip/clamp log written to: {args.log_path}")


if __name__ == "__main__":
    main()