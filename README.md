# nanodiffusion

Diffusion and Flow Matching from first principles. ~200 lines of PyTorch, runs on CPU.

**Goal**: Understand how modern generative models work by building the minimal version.

## Quick Start

```bash
cd ~/nanodiffusion
uv sync
uv run python nanodiffusion.py
```

This will:
1. Load 500 MNIST digits (16×16 grayscale)
2. Train with DDPM (100-step sampling)
3. Train with Flow Matching (4-step sampling)
4. Generate samples with both methods
5. Save comparison images

## What This Teaches

- **Unified U-Net** — same architecture for both diffusion and flow matching
- **DDPM** — Complex noise schedule, predict noise, stochastic sampling
- **Flow Matching** — Simple linear interpolation, predict velocity, deterministic sampling
- **Key insight**: Same model, different training targets

## The Core Difference (3 Things)

### 1. Training pairs

```python
# DDPM: complex schedule
x_noisy = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise

# Flow Matching: linear interpolation (t=0 is noise, t=1 is data)
x_t = (1 - t) * noise + t * x0
```

### 2. Training target

```python
# DDPM: predict noise
loss = mse(model(x_noisy, t), noise)

# Flow Matching: predict velocity (direction from noise to data)
velocity = x0 - noise
loss = mse(model(x_t, t), velocity)
```

### 3. Sampling

```python
# DDPM: 100 steps going backwards t=99 → t=0
for t in reversed(range(100)):
    pred = model(x, t)
    x = compute_mean(x, pred, t) + noise

# Flow Matching: 4 steps going forward t=0 → t=1 via Euler integration
for i in range(4):
    t = i / 4
    velocity = model(x, t)
    x = x + velocity * dt
```

## File Structure

```
nanodiffusion.py        # Complete implementation (~200 lines)
notebooks/
  01_intuition.ipynb    # Teaching companion
exp/                    # Optional: CIFAR-10 training (see below)
PLANNING.md             # Pedagogical decisions
README.md               # This file
pyproject.toml          # Dependencies
```

## CIFAR-10 Training (Optional)

For more serious training on CIFAR-10 with a proper U-Net:

```bash
cd ~/nanodiffusion/exp
uv pip install -r requirements.txt
uv run python train_cifar.py
```

This trains flow matching on 32×32 color images. Runs on MPS/CUDA.

## Architecture

```
Input: noisy image (16x16) + timestep
       ↓
Time embedding (sinusoidal → MLP)
       ↓
U-Net: ConvBlock (16x16) → downsample (7x7) → ConvBlock → upsample (16x16) → ConvBlock
       ↓
Output: predicted noise (DDPM) or velocity (Flow)
```

## Key Concepts

### Forward Process
- DDPM: Add noise via carefully designed schedule
- Flow: Linear interpolation between noise and data

### Reverse Process
- Both: Learn to predict (noise or velocity)
- DDPM: Subtract noise with stochastic steps (100 steps)
- Flow: Follow velocity field deterministically (4 steps)

### Training
Simple MSE loss on the predicted quantity.

## Why Both Methods?

| Aspect | DDPM | Flow Matching |
|--------|------|---------------|
| Sampling steps | 50-1000 | 1-4 |
| Training complexity | Higher | Lower |
| Modern relevance | SD 1.5/SDXL | Flux, SOTA |
| Teaching value | Historical context | Simpler, faster |

**Teaching strategy**: Learn the architecture once, see both training targets, understand why flow matching wins.

## Next Steps

- Add class conditioning
- Scale up to 64×64 color images
- Introduce cross-attention for text conditioning
- Classifier-free guidance

## Resources

- DDPM paper: https://arxiv.org/abs/2006.11239
- Flow Matching paper: https://arxiv.org/abs/2210.02747
- Rectified Flow: https://arxiv.org/abs/2209.03003
