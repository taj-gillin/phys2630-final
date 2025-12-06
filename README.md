# Anomalous Diffusion Inference

Deep learning and physics-informed approaches for inferring anomalous diffusion parameters (α, D₀) from particle trajectories.

## Overview

This project predicts diffusion parameters from 2D particle trajectories following fractional Brownian motion (fBM):

- **α (anomalous exponent)**: Characterizes diffusion type (α < 1: subdiffusion, α = 1: normal, α > 1: superdiffusion)
- **D₀ (diffusion coefficient)**: Characterizes diffusion magnitude

## Project Structure

```
├── configs/
│   ├── models/             # One config per model
│   │   ├── linear.yaml
│   │   ├── mlp.yaml
│   │   ├── lstm.yaml
│   │   ├── cnn.yaml
│   │   ├── hybrid.yaml
│   │   ├── hybrid_pinn.yaml
│   │   └── smoke.yaml      # Quick test
│   ├── data/
│   │   └── generate.yaml   # Data generation settings
│   └── compare.yaml        # Method comparison
│
├── scripts/
│   ├── train.py            # Train a single model
│   ├── run_comparison.py   # Compare methods
│   └── generate_data.py    # Generate & upload to HF
│
├── slurm/
│   └── submit.slurm        # Unified SLURM submission
│
├── src/
│   ├── data/               # Data loading
│   ├── methods/            # Model architectures
│   ├── evaluation/         # Metrics
│   └── utils/              # Config & logging
│
└── outputs/                # Trained models (by experiment name)
```

## Quick Start

### 1. Train a Model

Each model has its own config file. Train with:

```bash
# SLURM (recommended)
sbatch slurm/submit.slurm train configs/models/lstm.yaml

# Local
python scripts/train.py --config configs/models/lstm.yaml
```

### 2. Train All Models

```bash
for model in linear mlp lstm cnn hybrid hybrid_pinn; do
    sbatch slurm/submit.slurm train configs/models/${model}.yaml
done
```

### 3. Compare Methods

After training, compare all methods:

```bash
sbatch slurm/submit.slurm compare configs/compare.yaml
```

### 4. Quick Smoke Test

Verify everything works:

```bash
sbatch slurm/submit.slurm train configs/models/smoke.yaml
```

## Model Architectures

| Model | Encoder | Description |
|-------|---------|-------------|
| `linear` | Linear | Simple baseline |
| `mlp` | MLP | Multi-layer perceptron |
| `lstm` | BiLSTM | Bidirectional LSTM |
| `cnn` | CNN | Multi-scale 1D convolutions |
| `hybrid` | CNN+LSTM | Combined local & sequential |
| `hybrid_pinn` | CNN+LSTM | + Physics-informed loss |

## Configuration

Each config YAML has these sections:

```yaml
experiment:
  name: lstm                    # Used for output dir & wandb

wandb:
  enabled: true
  project: anomalous-diffusion
  entity: taj_gillin-Brown University
  tags: [lstm, recurrent]

data:
  repo_id: taj-gillin/andi-trajectory  # HuggingFace dataset
  n_train: 50000
  n_val: 5000

model:
  encoder: lstm
  loss: supervised
  params:
    hidden_dim: 64
    num_layers: 2

training:
  epochs: 50
  batch_size: 128
  lr: 0.001

Optional regularization settings live under the `training` block:

```yaml
training:
  epochs: 50
  batch_size: 128
  lr: 0.001
  regularization:
    l1: 1e-6
    l2: 1e-4
```
```

## Weights & Biases

All experiments log to wandb automatically:
- **Project**: `taj_gillin-Brown University/anomalous-diffusion`

Disable with:
```bash
WANDB_MODE=disabled sbatch slurm/submit.slurm train configs/models/lstm.yaml
```

Or in config:
```yaml
wandb:
  enabled: false
```

## Data

Training data comes from HuggingFace: [`taj-gillin/andi-trajectory`](https://huggingface.co/datasets/taj-gillin/andi-trajectory)

To regenerate/update the dataset:
```bash
sbatch slurm/submit.slurm generate configs/data/generate.yaml
```

## Outputs

Each experiment saves to `outputs/<experiment_name>/`:
- `best.pt` - Best model checkpoint
- `final.pt` - Final model checkpoint  
- `config.yaml` - Experiment config
- `summary.json` - Training history

## SLURM Usage

```bash
# Train a model
sbatch slurm/submit.slurm train configs/models/<model>.yaml

# Compare methods
sbatch slurm/submit.slurm compare configs/compare.yaml

# Generate data
sbatch slurm/submit.slurm generate configs/data/generate.yaml
```
