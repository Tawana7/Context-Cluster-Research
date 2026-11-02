r"""
make_flow_figure.py — publication-style figure: one CASME II sequence's onset / apex /
offset frames with the optical flow computed between them, plus a colour-wheel legend.

Run this on your PC (needs the Cropped frames; matplotlib + the project's OpenCV setup):

    python make_flow_figure.py ^
        --xlsx "C:\...\CASME2-coding-20140508.xlsx" ^
        --frames_root "C:\...\Cropped" ^
        --subject 17 --filename EP15_01 ^
        --flow_root "C:\...\Flow" ^
        --out flow_example

--flow_root (recommended): shows the EXACT flow images the model was trained on.
Without it, the flow is recomputed here with the same TV-L1 code.

Outputs <out>.pdf (vector, for LaTeX) and <out>.png (300 dpi).

PERMISSION: CASME II is distributed under a licence agreement. Before putting any
participant's face in a thesis or paper, check what that agreement allows, and prefer a
subject whose images the dataset authors themselves published (e.g. in the original CASME II
paper or in other MER papers you cite). Cite that source in the figure caption.
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.compute_flow_casme2 import compute_flow, frame_path, imread_unicode


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def color_wheel(size=256):
    """The same HSV encoding compute_flow_casme2.flow_to_rgb uses: hue = direction, brightness = magnitude."""
    ys, xs = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    mag = np.sqrt(xs ** 2 + ys ** 2)
    ang = np.arctan2(ys, xs) % (2 * np.pi)
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[mag > 1] = 255
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--filename", required=True)
    ap.add_argument("--flow_root", default=None, help="Use precomputed flow PNGs from here (recommended)")
    ap.add_argument("--method", default="tvl1", choices=["tvl1", "farneback"])
    ap.add_argument("--out", default="flow_example")
    ap.add_argument("--font", type=int, default=11)
    a = ap.parse_args()

    df = pd.read_excel(a.xlsx)
    row = df[(df["Subject"] == a.subject) & (df["Filename"].astype(str) == a.filename)]
    if row.empty:
        raise SystemExit(f"No row for Subject={a.subject}, Filename={a.filename}")
    row = row.iloc[0]
    onset, apex, offset = row["OnsetFrame"], row["ApexFrame"], row["OffsetFrame"]
    if apex == "/" or pd.isna(apex):
        apex = int((onset + offset) / 2)
    onset, apex, offset = int(onset), int(apex), int(offset)
    emotion = row.get("Estimated Emotion", "")
    aus = row.get("Action Units", "")

    frames = {}
    for name, idx in (("onset", onset), ("apex", apex), ("offset", offset)):
        img = imread_unicode(frame_path(a.frames_root, a.subject, a.filename, idx))
        if img is None:
            raise SystemExit(f"Could not read the {name} frame (index {idx}); check --frames_root.")
        frames[name] = img

    if a.flow_root:
        d = os.path.join(a.flow_root, f"sub{a.subject:02d}", a.filename)
        f_oa, f_ao = imread_unicode(os.path.join(d, "flow_onset_apex.png")), imread_unicode(os.path.join(d, "flow_apex_offset.png"))
        if f_oa is None or f_ao is None:
            raise SystemExit(f"No precomputed flow found in {d} (this sequence may have been skipped).")
        source = "flow images used for training"
    else:
        h, w = frames["onset"].shape[:2]
        f_oa = compute_flow(frames["onset"], cv2.resize(frames["apex"], (w, h)), a.method)
        f_ao = compute_flow(cv2.resize(frames["apex"], (w, h)), cv2.resize(frames["offset"], (w, h)), a.method)
        source = f"recomputed ({a.method})"

    plt.rcParams.update({"font.size": a.font, "font.family": "serif"})
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
    panels = [
        (axes[0, 0], bgr2rgb(frames["onset"]), f"Onset (frame {onset})"),
        (axes[0, 1], bgr2rgb(frames["apex"]), f"Apex (frame {apex})"),
        (axes[0, 2], bgr2rgb(frames["offset"]), f"Offset (frame {offset})"),
        (axes[1, 0], bgr2rgb(f_oa), "Flow: onset → apex"),
        (axes[1, 1], bgr2rgb(f_ao), "Flow: apex → offset"),
    ]
    for ax, img, title in panels:
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    axes[1, 2].axis("off")
    axes[0, 3].axis("off")
    axes[0, 3].text(0.0, 0.5, f"Subject {a.subject:02d}\n{a.filename}\n{emotion}\nAU {aus}",
                    va="center", ha="left", transform=axes[0, 3].transAxes, linespacing=1.6)
    wheel = axes[1, 3]
    wheel.imshow(color_wheel())
    wheel.set_title("Flow encoding")
    wheel.set_xticks([]); wheel.set_yticks([])
    for spine in wheel.spines.values():
        spine.set_visible(False)
    wheel.text(0.5, -0.06, "hue = direction\nbrightness = magnitude", ha="center", va="top",
               transform=wheel.transAxes, fontsize=a.font - 2)

    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=300)
    print(f"Wrote {a.out}.pdf and {a.out}.png  (flow source: {source})")
    print("Reminder: check the CASME II licence before publishing this subject's face, and cite where the image comes from.")


if __name__ == "__main__":
    main()
