# nanodiffusion Planning

## Pedagogical Philosophy

**Goal**: Teach generative modeling from first principles, accessible to high schoolers, no math intimidation.

**Inspiration**: Karpathy's nanoGPT — build intuition first, add complexity incrementally.

**Anti-pattern**: Umar Jamil's 5-hour Stable Diffusion course — math-first, code-dump, reference material disguised as tutorial.

---

## The Core Insight

**Diffusion and Flow Matching are the same thing with different parameterizations.**

They both:
1. Start with noise
2. Learn to predict something (noise or velocity)
3. Iteratively move toward clean data

The U-Net architecture is identical. Only the training target and sampling loop differ.

---

## Core Ratio

| Component | Percentage |
|-----------|------------|
| Core concepts (how it works) | 70% |
| Incremental complexity (make it bigger/faster) | 20% |
| Production details (DataLoader, GPU, etc.) | 10% |

---

## What to Include vs Skip

### Include (Core)
- **Unified U-Net** — same architecture for both approaches
- **Forward processes** — `q_sample_ddpm` vs `q_sample_flow`
- **Training targets** — predict noise vs predict velocity
- **Sampling loops** — DDPM vs Euler/ODE solver
- Time embeddings — sinusoidal positional encoding
- Training loop — simplified, no DataLoader initially
- **Shape printing at every step** — show tensor dimensions

### Include (Incremental)
- Cross-attention for conditioning
- Classifier-free guidance
- Scaling to larger images

### Skip for "Later" / Mention in Passing
- DataLoader and proper data pipelines
- Optimizers beyond Adam
- Advanced schedulers (cosine, sigmoid, DPMSolver, DDIM)
- Evaluation metrics (FID, IS)
- EMA, checkpointing, logging
- Mixed precision training
- Distributed training

---

## Lecture Structure

### Lecture 1: Core Intuition (30 min)
**Goal**: Understand that generative modeling = learn to move from noise to data.

```python
# Hardcoded data, no DataLoader
images = load_mnist()[:100]  # just 100 images
noise = torch.randn_like(images)

# The core idea: draw a line from noise to data
t = torch.rand(1)  # where are we on the line?
x_mixed = (1 - t) * noise + t * images

# Train model to predict direction
velocity = images - noise  # where to go?
model(x_mixed, t) → predict velocity
```

**Key insight**: The model learns "at this point on the noise→data line, which way should I go?"

### Lecture 2: The Architecture (30 min)
- Why U-Net? Encoder-decoder with skip connections
- Why time embeddings? Network needs to know "where are we on the line?"
- Why convolutions? Images are spatial
- Show what breaks without each component

### Lecture 3: Two Ways to Train the Same Model (30 min)
- **DDPM**: Complex noise schedule, predict noise, stochastic sampling
- **Flow Matching**: Linear interpolation, predict velocity, deterministic sampling
- Side-by-side code comparison
- Flow matching wins on speed (4 steps vs 100)

### Lecture 4: Conditioning (30 min)
- Start naive: concatenate class vector to spatial
- Show it fails
- Introduce cross-attention
- Side-by-side comparison

### Lecture 5: Scale Up (30 min)
- 16×16 → 64×64
- Grayscale → color
- CPU → GPU/MPS
- Mention: "Now we'd use DataLoader for streaming"

### Lecture 6: Text Conditioning (45 min)
- CLIP embeddings
- Classifier-free guidance
- Prompt engineering intuition
- Negative prompts

---

## Code Structure

**Main file**: `nanodiffusion.py`
- No DataLoader initially
- Hardcoded batch of 100 images
- Every tensor shape printed
- `mode='flow'` or `mode='ddpm'` parameter
- ~200 lines max

**Notebooks**: Teaching companion
- Concepts first, then code
- Visualizations of forward/reverse process
- Side-by-side comparisons

---

## The Core Difference (3 Things)

### 1. How you make training pairs

```python
# DDPM
sqrt_alpha_bar = alpha_bar[t].sqrt()
sqrt_one_minus = (1 - alpha_bar[t]).sqrt()
x_noisy = sqrt_alpha_bar * x0 + sqrt_one_minus * noise

# Flow Matching
t = torch.rand(batch, 1, 1, 1)
x_noisy = (1 - t) * noise + t * x0
```

### 2. What you predict

```python
# DDPM: predict the noise
loss = mse(model(x_noisy, t), noise)

# Flow Matching: predict the velocity
velocity = x0 - noise
loss = mse(model(x_noisy, t), velocity)
```

### 3. How you sample

```python
# DDPM: 100 steps, add noise at each step
for t in reversed(range(100)):
    pred = model(x, t)
    x = compute_mean(x, pred, t) + noise

# Flow Matching: 4 steps, follow the vector
for t in [0.75, 0.5, 0.25, 0.0]:
    velocity = model(x, t)
    x = x + velocity * dt
```

---

## Anti-Patterns to Avoid

1. **Math dump before code** — build intuition with tensors first
2. **Full architecture at once** — start with 1D signals, then 2D, then U-Net
3. **"It just works"** — show what breaks without each component
4. **Premature optimization** — DataLoader comes AFTER understanding the core
5. **Hidden abstractions** — every line should be explainable in 30 seconds
6. **DDPM vs Flow as separate topics** — they're the same model, different wrapper

---

## Reference vs Tutorial

| Reference | Tutorial |
|-----------|----------|
| "Here's the full code" | "Here's why we do this" |
| Copy-paste to use | Understand to modify |
| Covers edge cases | Covers core path |
| Assumes background | Teaches background |

**We're building a tutorial.**

---

## Why This Approach Works

1. **One U-Net to learn** — students see it once, understand it deeply
2. **Two training targets** — same architecture, different framing
3. **Flow matching is simpler** — students "get it" faster
4. **Diffusion provides context** — historical grounding, most tutorials use it
5. **Modern relevance** — Flow matching is what Flux and current SOTA use

---

## Next Steps

1. Rewrite `nanodiffusion.py` with unified model, both modes
2. Remove DataLoader, hardcode 100 images
3. Add shape print statements everywhere
4. Update notebook to show both approaches
5. Create comparison notebook: DDPM vs Flow side-by-side
