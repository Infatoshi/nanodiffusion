#!/usr/bin/env python
"""
Flow Matching on CIFAR-10 — local training on Apple MPS (M4 Max).

Perf budget: ~30-60 min for recognizable samples.
Architecture: proper U-Net with skip connections, residual blocks.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_EXP = Path(__file__).resolve().parent
_CIFAR_DIR = _EXP / "cifar10"

# ============== CONFIG ==============
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

IMG_SIZE = 32
BATCH_SIZE = 256
HIDDEN = 128
EMBED = 64
EPOCHS = 50
LR = 3e-4
SAVE_EVERY = 10
SAMPLE_STEPS = 8


# ============== TIMESTEP EMBEDDING ==============
class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0)) / half)
        )
        emb = t[:, None].float() * freq[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


# ============== U-NET BLOCKS ==============
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    3-level U-Net with skip connections.
    32x32 -> 16x16 -> 8x8 -> 16x16 -> 32x32
    """

    def __init__(self, in_ch=3, hidden=128, embed=64):
        super().__init__()
        self.time_embed = TimeEmbed(embed)

        self.enc1 = ResBlock(in_ch, hidden, embed)
        self.down1 = Downsample(hidden)
        self.enc2 = ResBlock(hidden, hidden * 2, embed)
        self.down2 = Downsample(hidden * 2)

        self.mid = ResBlock(hidden * 2, hidden * 2, embed)

        self.up2 = Upsample(hidden * 2)
        self.dec2 = ResBlock(hidden * 4, hidden, embed)
        self.up1 = Upsample(hidden)
        self.dec1 = ResBlock(hidden * 2, hidden, embed)

        self.out = nn.Conv2d(hidden, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        h1 = self.enc1(x, t_emb)
        h2 = self.enc2(self.down1(h1), t_emb)
        h = self.mid(self.down2(h2), t_emb)

        h = self.dec2(torch.cat([self.up2(h), h2], dim=1), t_emb)
        h = self.dec1(torch.cat([self.up1(h), h1], dim=1), t_emb)
        return self.out(h)


# ============== FLOW MATCHING ==============
def flow_loss(model, x0):
    """Conditional flow matching loss: predict velocity v = x1 - x0 along linear path."""
    noise = torch.randn_like(x0)
    t = torch.rand(x0.shape[0], device=x0.device)
    x_t = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * x0
    velocity = x0 - noise
    pred = model(x_t, t)
    return F.mse_loss(pred, velocity)


@torch.no_grad()
def sample(model, n=16, steps=SAMPLE_STEPS):
    model.eval()
    x = torch.randn(n, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i / steps, device=DEVICE)
        x = x + model(x, t) * dt
    return x.clamp(-1, 1)


# ============== DATA ==============
def get_loader():
    tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    ds = datasets.CIFAR10(str(_CIFAR_DIR), train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


# ============== SAVE ==============
def save_grid(samples, path, title=""):
    n = int(samples.shape[0] ** 0.5)
    fig, axes = plt.subplots(n, n, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < samples.shape[0]:
            img = (samples[i].cpu().permute(1, 2, 0).numpy() + 1) / 2
            ax.imshow(img.clip(0, 1))
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  saved {path}")


# ============== MAIN ==============
def main():
    print(f"device: {DEVICE}")

    samples_dir = _EXP / "samples"
    ckpt_dir = _EXP / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loader = get_loader()
    model = UNet(in_ch=3, hidden=HIDDEN, embed=EMBED).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss, n_batch = 0.0, 0

        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            loss = flow_loss(model, imgs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batch += 1

        elapsed = time.time() - t0
        avg = total_loss / n_batch
        print(f"epoch {epoch:3d} | loss {avg:.4f} | {elapsed:.0f}s")

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
            s = sample(model, n=16)
            save_grid(
                s,
                samples_dir / f"epoch_{epoch:03d}.png",
                f"epoch {epoch}",
            )
            torch.save(
                model.state_dict(),
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )

    total = time.time() - t0
    print(f"\ntotal: {total / 60:.1f} min")
    torch.save(model.state_dict(), ckpt_dir / "final.pt")

    s = sample(model, n=16)
    save_grid(s, samples_dir / "final.png", "final")


if __name__ == "__main__":
    main()
