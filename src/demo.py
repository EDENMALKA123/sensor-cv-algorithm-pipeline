"""Demo: classical preprocessing vs lightweight CNN denoiser on STL-10 grayscale."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import STL10
from torchvision.transforms import functional as TF

from . import classical
from .metrics import psnr_gray, ssim_gray
from .models import DenoiseCNN, restore_from_noise_prediction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--index", type=int, default=0, help="STL-10 test image index")
    p.add_argument("--noise-sigma", type=float, default=25.0)
    p.add_argument("--weights", type=str, default="", help="Optional path to weights.pt from train.py")
    p.add_argument("--out", type=str, default="./outputs/demo_compare.png")
    return p.parse_args()


def pil_rgb_to_gray_u8(img):
    t = TF.to_tensor(img)
    g = 0.2989 * t[0] + 0.5870 * t[1] + 0.1140 * t[2]
    return np.clip(g.numpy() * 255.0, 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = STL10(root=args.data_root, split="test", download=False)
    gray_u8 = pil_rgb_to_gray_u8(ds[args.index][0])

    clean_f = gray_u8.astype(np.float32)
    noise_np = np.random.randn(*clean_f.shape).astype(np.float32) * args.noise_sigma
    noisy_u8 = np.clip(clean_f + noise_np, 0, 255).astype(np.uint8)

    med = classical.median_blur_gray(noisy_u8, ksize=5)
    bil = classical.bilateral_gray(noisy_u8)
    analysis = classical.analyze_gray(noisy_u8)
    n_lab, labels = classical.connected_components(analysis.otsu_mask)
    cents = classical.label_centroids(labels, min_area=32)

    metrics_rows = []
    for name, recon in [
        ("noisy", noisy_u8),
        ("median", med),
        ("bilateral", bil),
    ]:
        metrics_rows.append(
            (name, psnr_gray(gray_u8, recon), ssim_gray(gray_u8, recon))
        )

    cnn_u8 = None
    if args.weights and Path(args.weights).is_file():
        try:
            ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.weights, map_location=device)
        cfg = ckpt.get("args", {})
        depth = int(cfg.get("depth", 8))
        width = int(cfg.get("width", 48))
        model = DenoiseCNN(depth=depth, width=width).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            ten = torch.from_numpy(noisy_u8.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            pred_noise = model(ten)
            restored = restore_from_noise_prediction(ten, pred_noise).squeeze().cpu().numpy()
            cnn_u8 = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
        metrics_rows.append(("cnn_residual", psnr_gray(gray_u8, cnn_u8), ssim_gray(gray_u8, cnn_u8)))

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.ravel()

    def show(ax, im, title):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    show(axes[0], gray_u8, "clean (gray)")
    show(axes[1], noisy_u8, f"noisy σ={args.noise_sigma}")
    show(axes[2], med, "median filter")
    show(axes[3], bil, "bilateral filter")

    show(axes[4], analysis.blurred_g, "gaussian blur")
    show(axes[5], analysis.sobel_mag, "sobel magnitude")
    show(axes[6], analysis.canny, "canny edges")
    show(axes[7], analysis.hist_eq, "histogram eq")

    show(axes[8], analysis.otsu_mask, f"otsu (t={analysis.otsu_thresh})")

    overlay = cv2.cvtColor(noisy_u8, cv2.COLOR_GRAY2RGB)
    for _, cx, cy in cents[:50]:
        cv2.circle(overlay, (int(cx), int(cy)), 3, (255, 0, 0), -1)
    axes[9].imshow(overlay)
    axes[9].set_title(f"connected components (centroids, n_labels={n_lab})")
    axes[9].axis("off")

    h0, w0 = noisy_u8.shape
    ph = min(64, h0)
    pw = min(64, w0)
    crop = noisy_u8[:ph, :pw]
    interp_demo = classical.resize_gray(crop, scale=4.0, interpolation="nearest")
    show(axes[10], interp_demo, "interp demo patch x4 nearest")

    if cnn_u8 is not None:
        show(axes[11], cnn_u8, "CNN residual denoise")
    else:
        axes[11].text(
            0.5,
            0.5,
            "Train model or pass --weights\nto compare CNN output.",
            ha="center",
            va="center",
        )
        axes[11].axis("off")

    plt.tight_layout()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outp, dpi=150)
    print(f"Saved figure to {outp.resolve()}")
    print("PSNR / SSIM vs clean:")
    for name, p, s in metrics_rows:
        print(f"  {name:14s}  PSNR={p:.2f} dB   SSIM={s:.3f}")


if __name__ == "__main__":
    main()
