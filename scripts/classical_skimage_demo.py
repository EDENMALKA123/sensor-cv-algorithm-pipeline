#!/usr/bin/env python3
"""Run classical IP/CV chain on skimage sample data (no PyTorch / STL-10 required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import classical  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="./outputs/classical_skimage_demo.png")
    args = p.parse_args()

    gray_u8 = (data.camera() >> 2).astype(np.uint8) * 4
    gray_u8 = np.clip(gray_u8, 0, 255)

    salt_pepper = gray_u8.copy()
    rng = np.random.default_rng(0)
    mask = rng.random(gray_u8.shape) < 0.02
    salt_pepper[mask] = rng.choice([0, 255], size=int(mask.sum()))

    analysis = classical.analyze_gray(salt_pepper)
    med = classical.median_blur_gray(salt_pepper, ksize=5)
    bil = classical.bilateral_gray(salt_pepper)
    edges = classical.canny_edges_gray(salt_pepper, low=40, high=120)
    lines = classical.hough_lines_p(edges, threshold=40, min_line_length=20, max_line_gap=8)
    er, dl = classical.morph_ops_gray(analysis.otsu_mask)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    titles_imgs = [
        ("original sample", gray_u8),
        ("salt & pepper", salt_pepper),
        ("median", med),
        ("bilateral", bil),
        ("hist eq", analysis.hist_eq),
        ("sobel mag", analysis.sobel_mag),
        ("otsu", analysis.otsu_mask),
        ("morph erode | dilate", np.hstack([er, dl])[:, : gray_u8.shape[1]]),
    ]
    for ax, (title, im) in zip(axes.ravel(), titles_imgs):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    import cv2

    overlay = np.stack([salt_pepper] * 3, axis=-1)
    for ln in lines[:80]:
        x1, y1, x2, y2 = ln[0].tolist()
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    ax2.imshow(overlay)
    ax2.set_title(f"Hough P lines (n={len(lines)})")
    ax2.axis("off")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=140, bbox_inches="tight")
    fig2.savefig(outp.with_name(outp.stem + "_hough.png"), dpi=140, bbox_inches="tight")
    print(f"Saved {outp.resolve()} and Hough overlay beside it.")


if __name__ == "__main__":
    main()
