# nanodiffusion - Current State

> Last updated: 2026-03-31

## What This Is

Minimal diffusion/flow matching implementation for teaching. Two methods:
- **DDPM**: Complex noise schedule, 100-step sampling
- **Flow Matching**: Linear interpolation, 4-step sampling (simpler, faster)

## Repository Status: SHIPPED ✓

- GitHub: https://github.com/infatoshi/nanodiffusion
- Clean, documented, ready for teaching
- Notebook at `notebooks/01_intuition.ipynb` has matplotlib visualizations

## Active Training Job on Anvil

**Location**: `~/nanovideo/` on `anvil` (RTX 3090, 24GB)

**What's training**: Video flow matching on colorful bouncing balls physics simulation
- 16 frames @ 48x48 RGB
- 2.6M param 3D UNet
- 30,000 steps
- ETA: ~2 hours from start (~04:45 UTC)

**Monitor**:
```bash
ssh anvil "tail -20 ~/nanovideo/train.log"
```

**Pull samples**:
```bash
scp anvil:~/nanovideo/outputs/*.gif .
```

**Files on anvil**:
- `~/nanovideo/train_physics.py` - training script
- `~/nanovideo/outputs/` - checkpoints, sample GIFs, loss plots
- `~/nanovideo/train.log` - live training output

## Next Steps

1. **Check training results** - Once done, pull GIFs and evaluate quality
2. **If good**: Create `nanovideo.py` - standalone video flow matching implementation for the repo
3. **If bad**: Tune hyperparameters or try different data (BAIR robot, etc.)
4. **Teaching goal**: Show video generation is just image diffusion + temporal dimension

## Key Decisions Made

- Using `uv` for package management (user preference)
- Flow Matching over DDPM for video (simpler, faster convergence)
- Synthetic physics data (no dataset download needed, clear visual feedback)
- 48x48 resolution balances quality vs training speed on 3090

## Commands Reference

```bash
# Check if training is running
ssh anvil "ps aux | grep train_physics | grep -v grep"

# View latest progress
ssh anvil "tail -30 ~/nanovideo/train.log"

# Kill training if needed
ssh anvil "pkill -f train_physics"

# Restart training
ssh anvil "cd ~/nanovideo; nohup uv run python -u train_physics.py > train.log 2>&1 &"

# Pull all outputs
scp -r anvil:~/nanovideo/outputs/ ./nanovideo-outputs/
```
