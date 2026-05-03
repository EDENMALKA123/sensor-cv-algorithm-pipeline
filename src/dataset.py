"""STL-10 based noisy patch dataset with augmentation (sensor-noise toy model)."""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import STL10


class RandomGaussianNoise:
    def __init__(self, sigma_min: float, sigma_max: float):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, clean01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        noise = torch.randn_like(clean01) * (sigma / 255.0)
        noisy = torch.clamp(clean01 + noise, 0.0, 1.0)
        return noisy, noise


class NoisyGrayPatchDataset(Dataset):
    """
    Pulls RGB STL-10 samples, converts to grayscale patches in [0,1].
    Target for the CNN is the additive noise tensor (sigma scaled like pixel intensities).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_size: int = 64,
        noise_min: float = 10.0,
        noise_max: float = 40.0,
        download: bool = False,
        augment: bool = True,
        subset: Optional[int] = None,
    ):
        assert split in {"train", "test"}
        self.patch_size = patch_size
        self.augment = augment and split == "train"
        self.noise = RandomGaussianNoise(noise_min, noise_max)
        self.ds = STL10(root=root, split=split, download=download)
        self.gray = transforms.Grayscale(num_output_channels=1)
        self.to_tensor = transforms.ToTensor()
        self.subset = subset

    def __len__(self) -> int:
        n = len(self.ds)
        return min(n, self.subset) if self.subset else n

    def _random_crop(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        ph = pw = self.patch_size
        if h < ph or w < pw:
            x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(ph, pw), mode="bilinear", align_corners=False).squeeze(0)
            return x
        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)
        return x[:, top : top + ph, left : left + pw]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.ds[idx]
        x = self.to_tensor(self.gray(img))
        if self.augment:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
            if random.random() < 0.5:
                x = torch.flip(x, dims=[1])
            k = random.randint(0, 3)
            if k:
                x = torch.rot90(x, k=k, dims=[1, 2])
        x = self._random_crop(x)
        noisy, noise = self.noise(x)
        return noisy, noise


def collate_identity(batch):
    noisy = torch.stack([b[0] for b in batch], dim=0)
    noise = torch.stack([b[1] for b in batch], dim=0)
    return noisy, noise
