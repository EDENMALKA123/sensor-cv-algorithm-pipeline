"""
CNN denoiser predicting additive Gaussian noise (DnCNN-style residual formulation).

Extension idea for interviews (transfer learning): replace the stem with the first 2–3 layers
of a torchvision ResNet pretrained on ImageNet, freeze early blocks, and train only later convs
for domain adaptation when moving from RGB natural images to sensor statistics.
"""

from __future__ import annotations

import torch
from torch import nn


class DenoiseCNN(nn.Module):
    """Lightweight fully-convolutional noise predictor for single-channel patches."""

    def __init__(self, depth: int = 8, width: int = 48):
        super().__init__()
        assert depth >= 2
        layers: list[nn.Module] = []
        layers.append(nn.Conv2d(1, width, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(width))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(width, 1, kernel_size=3, padding=1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        return self.net(noisy)


def restore_from_noise_prediction(noisy: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
    """Residual reconstruction: clean_hat = noisy - noise_hat (clamped to valid range)."""
    x = noisy - predicted_noise
    return torch.clamp(x, 0.0, 1.0)
