# nanodiffusion

> Diffusion and Flow Matching from first principles in ~300 lines of PyTorch

`nanodiffusion.py` trains both DDPM and Flow Matching on MNIST. No config files, no abstractions. Run it:

```bash
uv sync
uv run python nanodiffusion.py
```

You'll see loss curves, tensor shapes at every step, and two output images: `samples_ddpm.png` (100 steps) and `samples_flow.png` (4 steps). Same model architecture, different training targets.

I built this after realizing most diffusion codebases are practical before they're obvious. Papers are math-heavy, tutorials dump 500 lines of config, and you're left wondering what actually matters. This one strips it down to the core loop: mix noise with data, train a model to predict direction, sample by following that direction.

The notebook (`notebooks/01_intuition.ipynb`) goes deeper with matplotlib visualizations of the noise schedule, forward process, training curves, and sampling trajectory. It's designed to run top-to-bottom and produce all its own figures.

## The Core Insight

DDPM and Flow Matching are the same thing with different wrappers:

| | DDPM | Flow Matching |
|---|---|---|
| Forward | Complex schedule (α, β, cumsum) | Linear interpolation |
| Predicts | Noise ε | Velocity v = data - noise |
| Sampling | 100 stochastic steps | 4 Euler steps |

The U-Net is identical. Only the training target and sampling loop differ. Flow Matching wins on simplicity and speed.

## What To Look For

During training, watch the loss drop and samples improve:
- Early: pure noise, random pixels
- Middle: blurry digit-like shapes emerge
- Late: recognizable digits, though imperfect at 16x16

The printed tensor shapes show data flowing through the model. The 16x16 resolution and tiny U-Net are deliberate -- this runs on CPU in under a minute.

## How To Read The Code

Read `nanodiffusion.py` in this order:

1. Config and data loading
2. `SinusoidalEmbed` -- time conditioning (like positional encoding)
3. `ConvBlock` -- conv + norm + time injection
4. `TinyUNet` -- encode → bottleneck → decode
5. `q_sample_ddpm` vs `q_sample_flow` -- the forward processes
6. `train_ddpm` vs `train_flow` -- same loop, different targets
7. `sample_ddpm` vs `sample_flow` -- the reverse processes

If you understand those seven pieces, you understand diffusion.

## What's Not Here

- No VAE/latent space (we work in pixel space)
- No text conditioning or CLIP
- No classifier-free guidance
- No EMA, checkpointing, or fancy schedulers
- No distributed training

These matter for production. They don't matter for understanding.

## Next Steps

After this clicks, you can:
- Add class conditioning (concatenate one-hot to input)
- Scale to 64x64 color images
- Add cross-attention for text
- Study Flux/SD3 which use Flow Matching

## Related

If you want the GPT equivalent of this approach, see [yoctogpt](https://github.com/Infatoshi/yoctogpt) -- 100 lines, pure Python, no dependencies.

For GPU programming, I have a [CUDA course](https://www.youtube.com/playlist?list=PLxNPSjHT5qvvIlKHZ3CGUBphnJ4zhtzr_) (12 hrs, 500K+ views) that covers kernel programming from scratch.
