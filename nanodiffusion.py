import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# Auto-detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 16
TIMESTEPS = 100
HIDDEN_DIM = 128
EMBED_DIM = 32

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def load_data(n_samples=500):
    """Load MNIST data (lazy loading, not at import time)"""
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    full_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    images = torch.stack([full_data[i][0] for i in range(n_samples)])
    print(f"Loaded {images.shape[0]} images, shape: {images.shape}")
    print(
        f"  → [batch, channels, height, width] = [{images.shape[0]}, {images.shape[1]}, {images.shape[2]}, {images.shape[3]}]"
    )
    return images


class SinusoidalEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.fc2(F.silu(self.fc1(emb)))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = SinusoidalEmbed(EMBED_DIM)
        self.in_block = ConvBlock(1, HIDDEN_DIM, EMBED_DIM)
        self.down1 = nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM, 4, stride=2)
        self.mid_block = ConvBlock(HIDDEN_DIM, HIDDEN_DIM, EMBED_DIM)
        self.up1 = nn.ConvTranspose2d(HIDDEN_DIM, HIDDEN_DIM, 4, stride=2)
        self.out_block = ConvBlock(HIDDEN_DIM, HIDDEN_DIM, EMBED_DIM)
        self.final = nn.Conv2d(HIDDEN_DIM, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = self.in_block(x, t_emb)
        h = self.down1(h)
        h = self.mid_block(h, t_emb)
        h = self.up1(h)
        h = self.out_block(h, t_emb)
        return self.final(h)


def print_shapes(tag, **tensors):
    """Print tensor shapes for educational purposes"""
    parts = [f"{name}: {tuple(t.shape)}" for name, t in tensors.items()]
    print(f"  [{tag}] {', '.join(parts)}")


def q_sample_ddpm(x0, t, noise):
    """DDPM forward: add noise via complex schedule"""
    beta = torch.linspace(0.0001, 0.02, TIMESTEPS, device=x0.device)
    alpha = 1 - beta
    alpha_bar = alpha.cumprod(dim=0)
    sqrt_alpha_bar = alpha_bar[t][:, None, None, None].sqrt()
    sqrt_one_minus_alpha_bar = (1 - alpha_bar[t])[:, None, None, None].sqrt()
    return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise


def q_sample_flow(x0, noise):
    """Flow matching forward: linear interpolation

    t=0 → pure noise
    t=1 → clean data
    x_t = (1-t)*noise + t*data
    """
    t = torch.rand(x0.shape[0], 1, 1, 1, device=x0.device)
    x_t = (1 - t) * noise + t * x0
    return x_t, t.squeeze()


def train_ddpm(model, images, epochs=3, verbose_shapes=True):
    """Train with DDPM objective"""
    print("\n=== Training DDPM ===")
    print("Target: predict the noise that was added")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    images = images.to(DEVICE)

    start = time.time()
    for epoch in range(epochs):
        for i in range(20):
            idx = torch.randint(0, images.shape[0], (32,))
            x0 = images[idx]
            t = torch.randint(0, TIMESTEPS, (x0.shape[0],), device=x0.device)
            noise = torch.randn_like(x0)
            x_noisy = q_sample_ddpm(x0, t, noise)
            pred_noise = model(x_noisy, t.float())
            loss = F.mse_loss(pred_noise, noise)

            if verbose_shapes and epoch == 0 and i == 0:
                print_shapes("input", x0=x0, t=t, noise=noise)
                print_shapes("forward", x_noisy=x_noisy)
                print_shapes("output", pred_noise=pred_noise, target=noise)
                verbose_shapes = False

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 5 == 0:
                print(f"  epoch {epoch} step {i} loss {loss.item():.4f}")

    elapsed = time.time() - start
    print(f"Training time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    return model


def train_flow(model, images, epochs=3, verbose_shapes=True):
    """Train with Flow Matching objective"""
    print("\n=== Training Flow Matching ===")
    print("Target: predict velocity (direction from noise to data)")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    images = images.to(DEVICE)

    start = time.time()
    for epoch in range(epochs):
        for i in range(20):
            idx = torch.randint(0, images.shape[0], (32,))
            x0 = images[idx]
            noise = torch.randn_like(x0)
            x_t, t = q_sample_flow(x0, noise)
            velocity = x0 - noise
            pred_velocity = model(x_t, t)
            loss = F.mse_loss(pred_velocity, velocity)

            if verbose_shapes and epoch == 0 and i == 0:
                print_shapes("input", x0=x0, noise=noise, t=t)
                print_shapes("forward", x_t=x_t)
                print_shapes("target", velocity=velocity)
                print_shapes("output", pred_velocity=pred_velocity)
                verbose_shapes = False

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 5 == 0:
                print(f"  epoch {epoch} step {i} loss {loss.item():.4f}")

    elapsed = time.time() - start
    print(f"Training time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    return model


def sample_ddpm(model, shape=(4, 1, IMG_SIZE, IMG_SIZE), verbose=True):
    """DDPM sampling: 100 steps with stochastic denoising"""
    print(f"\n=== DDPM Sampling ({TIMESTEPS} steps) ===")
    model.eval()
    x = torch.randn(shape, device=DEVICE)
    beta = torch.linspace(0.0001, 0.02, TIMESTEPS, device=DEVICE)
    alpha = 1 - beta
    alpha_bar = alpha.cumprod(dim=0)

    if verbose:
        print_shapes("init", x=x)
        print(f"  Going from t={TIMESTEPS - 1} → t=0 (noise → data)")

    start = time.time()
    with torch.no_grad():
        for step, t_val in enumerate(reversed(range(TIMESTEPS))):
            t = torch.full((shape[0],), t_val, device=DEVICE, dtype=torch.float32)
            alpha_bar_t = alpha_bar[t_val].view(1, 1, 1, 1)
            beta_t = beta[t_val].view(1, 1, 1, 1)
            alpha_t = alpha[t_val].view(1, 1, 1, 1)
            sqrt_one_minus = (1 - alpha_bar_t).sqrt()
            sqrt_alpha = alpha_t.sqrt()

            pred_noise = model(x, t)
            mean = (x - beta_t / sqrt_one_minus * pred_noise) / sqrt_alpha

            if t_val > 0:
                alpha_bar_prev = alpha_bar[t_val - 1]
                var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean

    elapsed = time.time() - start
    if verbose:
        print_shapes("final", x=x)
    print(f"Sampling time: {elapsed:.2f}s")
    return x


def sample_flow(model, shape=(4, 1, IMG_SIZE, IMG_SIZE), steps=4, verbose=True):
    """Flow matching sampling: Euler integration from noise to data

    We start at t=0 (noise) and integrate toward t=1 (data).
    At each step, the model predicts velocity = direction toward data.
    """
    print(f"\n=== Flow Matching Sampling ({steps} steps) ===")
    model.eval()
    x = torch.randn(shape, device=DEVICE)
    dt = 1.0 / steps

    if verbose:
        print_shapes("init", x=x)
        print(f"  Going from t=0 → t=1 (noise → data) with dt={dt:.2f}")

    start = time.time()
    with torch.no_grad():
        for i in range(steps):
            t_val = i / steps
            t = torch.full((shape[0],), t_val, device=DEVICE)
            velocity = model(x, t)
            x = x + velocity * dt

            if verbose and i == 0:
                print_shapes(f"step {i}", t=t, velocity=velocity, x_new=x)

    elapsed = time.time() - start
    if verbose:
        print_shapes("final", x=x)
    print(f"Sampling time: {elapsed:.3f}s")
    return x


def save_samples(samples, filename):
    samples = samples.cpu().numpy()
    fig, axes = plt.subplots(1, samples.shape[0], figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(samples[i].squeeze(), cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("nanodiffusion: DDPM vs Flow Matching comparison")
    print("=" * 60)

    images = load_data(n_samples=500)

    model = TinyUNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: TinyUNet with {n_params:,} parameters")
    print("Architecture: input → down (16→7) → mid → up (7→16) → output")

    # Train and sample with DDPM
    model_ddpm = train_ddpm(model, images, epochs=2)
    samples_ddpm = sample_ddpm(model_ddpm)
    save_samples(samples_ddpm, "samples_ddpm.png")

    # Train and sample with Flow Matching (fresh model)
    model_flow = TinyUNet().to(DEVICE)
    model_flow = train_flow(model_flow, images, epochs=2)
    samples_flow = sample_flow(model_flow, steps=4)
    save_samples(samples_flow, "samples_flow.png")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"DDPM:  {TIMESTEPS} sampling steps → samples_ddpm.png")
    print("Flow:  4 sampling steps → samples_flow.png")
    print("\nKey difference: Same U-Net, different training targets!")
    print("  DDPM: predicts noise ε")
    print("  Flow: predicts velocity v = x_data - x_noise")
