# Sensor-Oriented Image Processing & CNN Denoising (End-to-End)

Python portfolio project aligned with **image sensor / ISP-style algorithm engineering**: classical preprocessing and analysis, plus a **deep learning denoiser** trained on realistic additive noise. Suitable to discuss for roles mixing **image processing**, **computer vision**, and **deep learning** (mobile/automotive sensor context).

## What this repo demonstrates

| Stage | Topics |
|--------|--------|
| **Data** | STL-10 download via `torchvision`, patch extraction, **Gaussian noise simulation** (sensor read noise simplified), **data augmentation** (flips, rotations, crops). |
| **Classical IP/CV** | Spatial **convolution**, **Gaussian / median / bilateral** filtering, **Sobel / Canny** edges, **thresholding** (global, adaptive, Otsu), **histogram equalization**, **morphology**, **connected components**, **probabilistic Hough** lines, **resize interpolation** (nearest / bilinear / bicubic). |
| **Deep learning** | **CNN** with conv layers + **BatchNorm** + ReLU, **residual / noise prediction** (DnCNN-style idea), **training loop**, **metrics** (PSNR / SSIM). Optional extension: swap in a **torchvision** backbone for **transfer learning** (see code comments in `models.py`). |

## Inspired-by references (industry / research style)

- Joint demosaic/denoise/SR and raw pipelines: [End-to-End-JDNDMSR](https://github.com/xingwz/End-to-End-JDNDMSR), blind raw denoising [YOND_public](https://github.com/fenghansen/YOND_public).
- Differentiable imaging stacks: [End2endImaging](https://github.com/vccimaging/End2endImaging).
- Educational CNN denoising: [DnCNN](https://github.com/cszn/DnCNN), [UNet-Image-Denoising](https://github.com/CodeKnight314/UNet-Image-Denoising).

## Setup

```bash
cd sensor-cv-algorithm-pipeline
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Use the official CUDA build from [pytorch.org](https://pytorch.org/) instead of the CPU wheel if you want GPU training.

## Prepare data

Downloads STL-10 into `./data` (first run may take a few minutes):

```bash
python scripts/prepare_data.py --root ./data
```

## Train the denoiser (CNN)

Example (CPU-friendly small run):

```bash
python -m src.train --data-root ./data --epochs 5 --batch-size 16 --patch-size 64 --noise-min 10 --noise-max 40 --out-dir ./outputs
```

Weights are saved under `./outputs/weights.pt`.

## Classical pipeline + comparison demo

Runs classical algorithms on a sample image and compares **median / bilateral / CNN** (if weights exist):

```bash
python -m src.demo --data-root ./data --weights ./outputs/weights.pt
```

Omit `--weights` to run classical-only visualization.

### Classical-only quick demo (no STL-10 download)

Uses `skimage.data.camera` plus synthetic salt-and-pepper noise:

```bash
pip install numpy opencv-python-headless scikit-image scipy matplotlib
python scripts/classical_skimage_demo.py
```

## Interview angles

- **Why denoising:** Maps directly to sensor **SNR**, low-light behavior, and ISP tuning—easy to justify for Samsung Sensor-style teams.
- **Classical vs DL:** Trade-offs (latency on device, interpretability of kernels vs learned filters, edge preservation with bilateral vs CNN).
- **Augmentation:** Models robustness to orientation and crops—standard for mobile capture variability.

## Repository layout

```
sensor-cv-algorithm-pipeline/
  README.md
  requirements.txt
  docs/index.html   # portfolio / GitHub Pages overview of the whole project
  scripts/prepare_data.py
  src/
    classical.py      # classical IP/CV chain + utilities
    dataset.py        # noisy patches + augmentations
    models.py         # DenoiseCNN (noise prediction)
    metrics.py        # PSNR, SSIM
    train.py          # training CLI
    demo.py           # visualization & benchmarking
```

## Static HTML (GitHub Pages)

The full project summary for recruiters and reviewers lives in [**docs/index.html**](docs/index.html). After you push to GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch: `main` → Folder: `/docs`**. The site URL will be `https://edenmalka123.github.io/<repo-name>/` (use your exact repo name in the path).

