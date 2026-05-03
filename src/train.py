"""Train residual Gaussian denoiser on STL-10 patches."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import NoisyGrayPatchDataset, collate_identity
from .models import DenoiseCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--noise-min", type=float, default=10.0)
    p.add_argument("--noise-max", type=float, default=40.0)
    p.add_argument("--subset", type=int, default=0, help="Limit train images for quick experiments (0=all)")
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="./outputs")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    subset = args.subset if args.subset > 0 else None
    train_ds = NoisyGrayPatchDataset(
        root=args.data_root,
        split="train",
        patch_size=args.patch_size,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        download=False,
        augment=True,
        subset=subset,
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_identity,
    )

    model = DenoiseCNN(depth=args.depth, width=args.width).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for noisy, noise in pbar:
            noisy = noisy.to(device)
            noise = noise.to(device)
            pred = model(noisy)
            loss = loss_fn(pred, noise)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += loss.item() * noisy.size(0)
            n += noisy.size(0)
            pbar.set_postfix(loss=running / max(n, 1))
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
            },
            out / "weights.pt",
        )


if __name__ == "__main__":
    main()
