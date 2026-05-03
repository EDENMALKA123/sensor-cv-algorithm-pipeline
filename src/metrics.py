"""Image quality metrics for denoising evaluation."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def to_float01(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    return x.astype(np.float32)


def psnr_gray(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    return float(peak_signal_noise_ratio(a, b, data_range=data_range))


def ssim_gray(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    return float(structural_similarity(a, b, data_range=data_range))
