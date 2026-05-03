#!/usr/bin/env python3
"""
Regenerate figures under docs/assets/ for the GitHub Pages walkthrough.
Run from repo root: python scripts/export_docs_assets.py
Requires: numpy, opencv-python-headless, scikit-image, scipy, matplotlib (same as classical demo).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import classical  # noqa: E402
from src.metrics import psnr_gray, ssim_gray  # noqa: E402


def ensure_assets_dir() -> Path:
    d = ROOT / "docs" / "assets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    import cv2

    assets = ensure_assets_dir()
    rng = np.random.default_rng(42)

    gray_u8 = np.clip((data.camera() >> 2).astype(np.uint8) * 4, 0, 255)

    # Salt & pepper (matches classical_skimage_demo narrative)
    salt_pepper = gray_u8.copy()
    mask = rng.random(gray_u8.shape) < 0.02
    salt_pepper[mask] = rng.choice([0, 255], size=int(mask.sum()))

    analysis = classical.analyze_gray(salt_pepper)
    med_sp = classical.median_blur_gray(salt_pepper, ksize=5)
    bil_sp = classical.bilateral_gray(salt_pepper)
    er, dl = classical.morph_ops_gray(analysis.otsu_mask)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    fig.suptitle("Classical pipeline (skimage camera + salt & pepper)", fontsize=12, color="#dbe8ff")
    titles_imgs = [
        ("Clean reference", gray_u8),
        ("Salt & pepper input", salt_pepper),
        ("Median filter", med_sp),
        ("Bilateral filter", bil_sp),
        ("Histogram equalization", analysis.hist_eq),
        ("Sobel magnitude", analysis.sobel_mag),
        ("Otsu mask", analysis.otsu_mask),
        ("Morphology: erode | dilate", np.hstack([er, dl])[:, : gray_u8.shape[1]]),
    ]
    for ax, (title, im) in zip(axes.ravel(), titles_imgs):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    grid_path = assets / "classical_pipeline_grid.png"
    fig.savefig(grid_path, dpi=160, bbox_inches="tight", facecolor="#1a2332")
    plt.close(fig)

    edges = classical.canny_edges_gray(salt_pepper, low=40, high=120)
    lines = classical.hough_lines_p(edges, threshold=40, min_line_length=20, max_line_gap=8)
    overlay = np.stack([salt_pepper] * 3, axis=-1)
    for ln in lines[:80]:
        x1, y1, x2, y2 = ln[0].tolist()
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 64, 64), 1, cv2.LINE_AA)
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7))
    ax2.imshow(overlay)
    ax2.set_title(f"Canny edges → probabilistic Hough lines (n={len(lines)})", fontsize=11, color="#dbe8ff")
    ax2.axis("off")
    hough_path = assets / "hough_lines_overlay.png"
    fig2.patch.set_facecolor("#1a2332")
    fig2.savefig(hough_path, dpi=160, bbox_inches="tight", facecolor="#1a2332")
    plt.close(fig2)

    # Gaussian noise (closer to CNN training setup) + metric bars
    sigma = 25.0
    noisy_g = np.clip(gray_u8.astype(np.float32) + rng.standard_normal(gray_u8.shape).astype(np.float32) * sigma, 0, 255).astype(np.uint8)
    med_g = classical.median_blur_gray(noisy_g, ksize=5)
    bil_g = classical.bilateral_gray(noisy_g)

    fig3, axes3 = plt.subplots(1, 4, figsize=(14, 3.8))
    fig3.suptitle(f"Gaussian noise σ={sigma:.0f} (sensor-style read-noise toy model)", fontsize=11, color="#dbe8ff")
    row = [
        ("Clean", gray_u8),
        ("Noisy", noisy_g),
        ("Median", med_g),
        ("Bilateral", bil_g),
    ]
    for ax, (title, im) in zip(axes3, row):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig3.tight_layout()
    gauss_path = assets / "gaussian_denoise_compare.png"
    fig3.savefig(gauss_path, dpi=160, bbox_inches="tight", facecolor="#1a2332")
    plt.close(fig3)

    labels = ["Noisy", "Median", "Bilateral"]
    recon = [noisy_g, med_g, bil_g]
    psnrs = [psnr_gray(gray_u8, r) for r in recon]
    ssims = [ssim_gray(gray_u8, r) for r in recon]

    fig4, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))
    fig4.suptitle("Quality vs clean reference (PSNR ↑ , SSIM ↑ )", fontsize=11, color="#dbe8ff")
    x = np.arange(len(labels))
    ax_a.bar(x, psnrs, color=["#4d79a8", "#f28e2b", "#59a14f"])
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("PSNR (dB)")
    ax_a.set_facecolor("#121922")
    ax_a.grid(axis="y", alpha=0.25)

    ax_b.bar(x, ssims, color=["#4d79a8", "#f28e2b", "#59a14f"])
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("SSIM")
    ax_b.set_ylim(0, 1)
    ax_b.set_facecolor("#121922")
    ax_b.grid(axis="y", alpha=0.25)

    for ax in (ax_a, ax_b):
        ax.tick_params(colors="#c8d4e8")
        ax.yaxis.label.set_color("#c8d4e8")
        for spine in ax.spines.values():
            spine.set_color("#2c3a50")

    fig4.patch.set_facecolor("#1a2332")
    fig4.tight_layout()
    metrics_path = assets / "metrics_bars.png"
    fig4.savefig(metrics_path, dpi=160, bbox_inches="tight", facecolor="#1a2332")
    plt.close(fig4)

    # Tiny JSON snippet for optional future JS consumption
    summary_path = assets / "run_summary.json"
    summary_path.write_text(
        '{"figures":["classical_pipeline_grid.png","hough_lines_overlay.png","gaussian_denoise_compare.png","metrics_bars.png"],'
        f'"psnr_vs_clean_gaussian":{{"noisy":{psnrs[0]:.4f},"median":{psnrs[1]:.4f},"bilateral":{psnrs[2]:.4f}}},'
        f'"ssim_vs_clean_gaussian":{{"noisy":{ssims[0]:.4f},"median":{ssims[1]:.4f},"bilateral":{ssims[2]:.4f}}}'
        "}",
        encoding="utf-8",
    )

    print("Wrote:")
    for p in (grid_path, hough_path, gauss_path, metrics_path, summary_path):
        print(" ", p.resolve())


if __name__ == "__main__":
    main()
