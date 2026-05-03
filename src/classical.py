"""
Classical image processing building blocks relevant to sensor / ISP-style pipelines.

Uses OpenCV + NumPy/SciPy; documents interpolation modes and common preprocess stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d

Interpolation = Literal["nearest", "bilinear", "bicubic"]


def convolve_gray(image_u8: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution on single-channel uint8 image (reflect padding)."""
    x = image_u8.astype(np.float32) / 255.0
    k = kernel.astype(np.float32)
    if k.sum() != 0:
        k = k / k.sum()
    y = convolve2d(x, k, mode="same", boundary="symm")
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    """Discrete 2D Gaussian kernel (odd size)."""
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def gaussian_blur_gray(image_u8: np.ndarray, ksize: int = 5, sigma: float = 1.2) -> np.ndarray:
    ksize = int(ksize) | 1  # force odd
    k = gaussian_kernel_2d(ksize, sigma)
    return convolve_gray(image_u8, k)


def median_blur_gray(image_u8: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.medianBlur(image_u8, ksize)


def bilateral_gray(image_u8: np.ndarray, d: int = 7, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    return cv2.bilateralFilter(image_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def sobel_edges_gray(image_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(image_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = np.clip(mag, 0, 255).astype(np.uint8)
    gx_u8 = cv2.convertScaleAbs(gx)
    gy_u8 = cv2.convertScaleAbs(gy)
    return gx_u8, gy_u8, mag_u8


def canny_edges_gray(image_u8: np.ndarray, low: float = 50.0, high: float = 150.0) -> np.ndarray:
    return cv2.Canny(image_u8, threshold1=low, threshold2=high)


def threshold_simple_gray(image_u8: np.ndarray, thresh: int = 127, invert: bool = False) -> np.ndarray:
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, bw = cv2.threshold(image_u8, thresh, 255, mode)
    return bw


def threshold_adaptive_gray(image_u8: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    block_size = int(block_size) | 1
    return cv2.adaptiveThreshold(
        image_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
    )


def threshold_otsu_gray(image_u8: np.ndarray, invert: bool = False) -> Tuple[np.ndarray, int]:
    mode = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU if invert else cv2.THRESH_BINARY | cv2.THRESH_OTSU
    t, bw = cv2.threshold(image_u8, 0, 255, mode)
    return bw, int(t)


def histogram_equalization_gray(image_u8: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(image_u8)


def morph_ops_gray(bin_u8: np.ndarray, erode_ksize: int = 3, dilate_ksize: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    e_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_ksize, erode_ksize))
    d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ksize, dilate_ksize))
    eroded = cv2.erode(bin_u8, e_kernel)
    dilated = cv2.dilate(bin_u8, d_kernel)
    return eroded, dilated


def connected_components(bin_u8: np.ndarray) -> Tuple[int, np.ndarray]:
    """Returns (num_labels, labels_int32) for binary image."""
    n, labels = cv2.connectedComponents((bin_u8 > 0).astype(np.uint8))
    return int(n), labels


def hough_lines_p(
    edges_u8: np.ndarray,
    rho: float = 1.0,
    theta_deg: float = 1.0,
    threshold: int = 50,
    min_line_length: int = 30,
    max_line_gap: int = 10,
) -> np.ndarray:
    lines = cv2.HoughLinesP(
        edges_u8,
        rho,
        np.deg2rad(theta_deg),
        threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return np.empty((0, 1, 4), dtype=np.int32)
    return lines


def resize_gray(image_u8: np.ndarray, scale: float, interpolation: Interpolation) -> np.ndarray:
    h, w = image_u8.shape[:2]
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    interp = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }[interpolation]
    return cv2.resize(image_u8, (nw, nh), interpolation=interp)


@dataclass
class ClassicalAnalysis:
    blurred_g: np.ndarray
    median: np.ndarray
    bilateral: np.ndarray
    sobel_mag: np.ndarray
    canny: np.ndarray
    otsu_mask: np.ndarray
    otsu_thresh: int
    hist_eq: np.ndarray


def analyze_gray(image_u8: np.ndarray) -> ClassicalAnalysis:
    """Run a compact classical chain for demos / visualization."""
    blurred = gaussian_blur_gray(image_u8, ksize=5, sigma=1.2)
    med = median_blur_gray(image_u8, ksize=5)
    bil = bilateral_gray(image_u8)
    _, _, smag = sobel_edges_gray(blurred)
    edges = canny_edges_gray(blurred, low=40, high=120)
    otsu_mask, t = threshold_otsu_gray(blurred, invert=False)
    hist = histogram_equalization_gray(image_u8)
    return ClassicalAnalysis(
        blurred_g=blurred,
        median=med,
        bilateral=bil,
        sobel_mag=smag,
        canny=edges,
        otsu_mask=otsu_mask,
        otsu_thresh=t,
        hist_eq=hist,
    )


def label_centroids(labels: np.ndarray, min_area: int = 64) -> list[tuple[int, float, float]]:
    """Return [(label_id, cx, cy), ...] for components above min_area (excluding background 0)."""
    out: list[tuple[int, float, float]] = []
    max_label = int(labels.max())
    for lab in range(1, max_label + 1):
        mask = labels == lab
        area = int(mask.sum())
        if area < min_area:
            continue
        cy, cx = ndimage.center_of_mass(mask)
        out.append((lab, float(cx), float(cy)))
    return out
