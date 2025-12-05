#!/bin/bash
#SBATCH --job-name=train_generalized
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Train generalized models (LSTM, CNN, PINN)
# Usage: sbatch slurm/train_models.slurm [--config CONFIG_PATH]

echo "=============================================="
echo "Generalized Model Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Project root: $(pwd)"

# Parse arguments
CONFIG="${1:-experiments/exp_train_models.yaml}"
echo "Config: $CONFIG"
echo "Start time: $(date)"

# Activate environment
source ~/miniconda3/bin/activate
echo ""
echo "Python: $(which python)"
python --version

# Check for GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Run training
echo ""
echo "Starting training..."
python scripts/train_models.py --config "$CONFIG"

echo ""
echo "End time: $(date)"
echo "=============================================="

