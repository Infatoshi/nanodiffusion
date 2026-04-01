#!/usr/bin/env python
"""
Quick benchmarking for training on MPS / CUDA / CPU.

Usage (from repo root):
    uv run python exp/bench.py
    cd exp && python bench.py
"""

import argparse
import sys
import time
from pathlib import Path

_exp = Path(__file__).resolve().parent
if str(_exp) not in sys.path:
    sys.path.insert(0, str(_exp))

import torch
import torch.nn.functional as F

from train_cifar import BATCH_SIZE, DEVICE, EMBED, HIDDEN, IMG_SIZE, UNet


def bench(steps=20):
    model = UNet(in_ch=3, hidden=HIDDEN, embed=EMBED).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    x = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    for _ in range(3):
        noise = torch.randn_like(x)
        t = torch.rand(BATCH_SIZE, device=DEVICE)
        x_t = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * x
        pred = model(x_t, t)
        loss = F.mse_loss(pred, x - noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()

    t0 = time.time()
    for _step in range(steps):
        noise = torch.randn_like(x)
        t = torch.rand(BATCH_SIZE, device=DEVICE)
        x_t = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * x
        pred = model(x_t, t)
        loss = F.mse_loss(pred, x - noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()

    elapsed = time.time() - t0
    ms_per_step = elapsed / steps * 1000
    imgs_per_sec = BATCH_SIZE * steps / elapsed

    print(f"{'=' * 50}")
    print(f"  device:         {DEVICE}")
    print(f"  params:         {n_params:,}")
    print(f"  batch_size:     {BATCH_SIZE}")
    print(f"  img_size:       {IMG_SIZE}")
    print(f"  steps:          {steps}")
    print(f"  ms/step:        {ms_per_step:.1f}")
    print(f"  imgs/sec:       {imgs_per_sec:.0f}")
    print(f"  est epoch:      {50000 / imgs_per_sec:.1f}s (CIFAR-10, 50k imgs)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    bench(steps=args.steps)
