#!/usr/bin/env python3
"""Download STL-10 into ./data for the denoising pipeline."""

from __future__ import annotations

import argparse

from torchvision.datasets import STL10


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="./data")
    args = p.parse_args()
    STL10(root=args.root, split="train", download=True)
    STL10(root=args.root, split="test", download=True)
    print(f"STL-10 ready under {args.root}")


if __name__ == "__main__":
    main()
