# Anomalous Diffusion Inference

Deep learning and physics-informed approaches for inferring anomalous diffusion parameters from particle trajectories, following the [ANDI Challenge](https://www.nature.com/articles/s41467-021-26320-w) framework.

## Overview

This project predicts diffusion parameters from 2D particle trajectories:

**Task 1: α Inference**
- **α (anomalous exponent)**: Characterizes diffusion type (α < 1: subdiffusion, α = 1: normal, α > 1: superdiffusion)
- Primary metric: Mean Absolute Error (MAE)

**Task 2: Model Classification**
- Classify among 5 theoretical diffusion models: CTRW, FBM, LW, ATTM, SBM
- Primary metric: Micro-averaged F1-score

## Diffusion Models

Following the ANDI Challenge convention:

| Index | Model | Description | α Range |
|-------|-------|-------------|---------|
| 0 | **CTRW** | Continuous-Time Random Walk | 0.1 - 1.0 |
| 1 | **FBM** | Fractional Brownian Motion | 0.1 - 2.0 |
| 2 | **LW** | Lévy Walk | 1.0 - 2.0 |
| 3 | **ATTM** | Annealed Transient Time Motion | 0.1 - 1.0 |
| 4 | **SBM** | Scaled Brownian Motion | 0.1 - 2.0 |

## Project Structure

```
├── configs/
│   ├── models/                # One config per model
│   │   ├── lstm.yaml          # Task 1 only (α inference)
│   │   ├── lstm_multitask.yaml # Task 1 + Task 2 (multi-task)
│   │   └── ...
│   ├── data/
│   │   └── generate.yaml      # Data generation settings
│   └── compare.yaml           # Method comparison
│
├── scripts/
│   ├── train.py               # Train a single model
│   ├── run_comparison.py      # Compare methods
│   ├── generate_data.py       # Generate & upload to HF
│   └── analyze_results.py     # Extended analysis
│
├── slurm/
│   └── submit.slurm           # Unified SLURM submission
│
├── src/
│   ├── data/                  # Data loading & generation
│   │   ├── trajectory.py      # Trajectory class with model labels
│   │   ├── synthetic.py       # Multi-model trajectory generation
│   │   └── hf_loader.py       # HuggingFace dataset loading
│   ├── methods/               # Model architectures
│   │   ├── predictor.py       # Multi-task predictor (α + model)
│   │   ├── losses.py          # MSE, CrossEntropy, MultiTask losses
│   │   └── encoders.py        # Linear, MLP, LSTM, CNN, Hybrid
│   ├── evaluation/            # Metrics
│   │   └── metrics.py         # MAE, F1, confusion matrix, etc.
│   └── utils/                 # Config & logging
│
└── outputs/                   # Trained models (by experiment name)
```

## Quick Start

### 1. Train a Model (Task 1 Only)

```bash
# SLURM (recommended)
sbatch slurm/submit.slurm train configs/models/lstm.yaml

# Local
python scripts/train.py --config configs/models/lstm.yaml
```

### 2. Train Multi-Task Model (Task 1 + Task 2)

```bash
sbatch slurm/submit.slurm train configs/models/lstm_multitask.yaml
```

### 3. Generate Multi-Model Dataset

Generate trajectories with all 5 diffusion models:

```bash
sbatch slurm/submit.slurm generate configs/data/generate.yaml
```

### 4. Analyze Results

Generate detailed analysis with ANDI-style metrics:

```bash
# Analyze a single model
sbatch slurm/submit.slurm analyze outputs/lstm/best.pt

# Analyze and compare all models
sbatch slurm/submit.slurm analyze-all
```

## Model Architectures

| Model | Encoder | Description | Tasks |
|-------|---------|-------------|-------|
| `linear` | Linear | Simple baseline | α only |
| `mlp` | MLP | Multi-layer perceptron | α only |
| `lstm` | BiLSTM | Bidirectional LSTM | α only |
| `cnn` | CNN | Multi-scale 1D convolutions | α only |
| `hybrid` | CNN+LSTM | Combined local & sequential | α only |
| `lstm_multitask` | BiLSTM | Multi-task learning | α + model |

## Configuration

### Single-Task Config (Task 1 only)

```yaml
experiment:
  name: lstm

model:
  encoder: lstm
  loss: supervised
  # tasks: [alpha]  # Default
  params:
    hidden_dim: 64
    num_layers: 2
```

### Multi-Task Config (Task 1 + Task 2)

```yaml
experiment:
  name: lstm-multitask

model:
  encoder: lstm
  tasks: [alpha, model]  # Enable both tasks
  loss: multitask
  loss_params:
    lambda_alpha: 1.0    # Weight for α MSE
    lambda_model: 1.0    # Weight for model CrossEntropy
  params:
    hidden_dim: 64
    num_layers: 2
```

### Data Generation Config

```yaml
generation:
  n_trajectories: 200000
  models: [CTRW, FBM, LW, ATTM, SBM]  # All 5 models
  alpha_range: [0.1, 2.0]
  length_range: [50, 1000]
  snr_levels: null  # Or [1, 2, 10] for localization noise
  balanced_models: true
```

## Metrics (ANDI Challenge Comparison)

### Task 1: α Inference

| Metric | Description | ANDI Reference |
|--------|-------------|----------------|
| **MAE** | Mean Absolute Error | Primary metric |
| **RMSE** | Root Mean Squared Error | - |
| **Bias** | Mean(α_pred - α_true) | Should be ~0 |
| **R²** | Coefficient of determination | - |
| **P90/P95** | 90th/95th percentile error | Worst-case |

### Task 2: Model Classification

| Metric | Description | ANDI Reference |
|--------|-------------|----------------|
| **F1 (micro)** | Micro-averaged F1-score | Primary metric |
| **F1 (macro)** | Macro-averaged F1-score | - |
| **Accuracy** | Overall classification accuracy | - |
| **Confusion Matrix** | Per-class breakdown | - |

### Breakdown Analysis

Following ANDI methodology:
- **By α value**: subdiffusion (α<1) vs normal (α≈1) vs superdiffusion (α>1)
- **By trajectory length**: short (<100) to long (>600 points)
- **By model type**: Per-class performance for Task 2

## SLURM Usage

```bash
# Train a model
sbatch slurm/submit.slurm train configs/models/<model>.yaml

# Compare methods
sbatch slurm/submit.slurm compare configs/compare.yaml

# Generate data
sbatch slurm/submit.slurm generate configs/data/generate.yaml

# Analyze single model
sbatch slurm/submit.slurm analyze outputs/<model>/best.pt

# Compare all models
sbatch slurm/submit.slurm analyze-all
```

## Outputs

Each experiment saves to `outputs/<experiment_name>/`:
- `best.pt` - Best model checkpoint
- `final.pt` - Final model checkpoint  
- `config.yaml` - Experiment config
- `summary.json` - Training history
- `analysis/` - Extended analysis
  - `analysis_results.json` - Full metrics (Task 1 + Task 2)
  - `scatter_pred_vs_true.png` - α prediction scatter plot
  - `confusion_matrix.png` - Model classification confusion matrix
  - `bias_histogram.png` - Bias distribution
  - `error_vs_alpha.png` - Error by α value
  - `error_vs_length.png` - Error by trajectory length

## Data

Training data comes from HuggingFace: [`taj-gillin/andi-trajectory`](https://huggingface.co/datasets/taj-gillin/andi-trajectory)

Features per trajectory:
- `x`, `y`: Position coordinates
- `alpha`: Anomalous diffusion exponent
- `model`: Diffusion model index (0-4)
- `model_name`: Model name (CTRW, FBM, LW, ATTM, SBM)
- `length`: Trajectory length
- `snr`: Signal-to-noise ratio (if noise added)

## References

- [ANDI Challenge Paper](https://www.nature.com/articles/s41467-021-26320-w): Muñoz-Gil et al., "Objective comparison of methods to decode anomalous diffusion", Nature Communications (2021)
- [andi-datasets Library](https://github.com/AnDiChallenge/ANDI_datasets): Official dataset generation tools
