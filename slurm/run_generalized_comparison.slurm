#!/bin/bash
#SBATCH --job-name=generalized_comparison
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Compare generalized models against baselines
# Usage: sbatch slurm/run_generalized_comparison.slurm

echo "=============================================="
echo "Generalized Model Comparison"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Project root: $(pwd)"

CONFIG="${1:-experiments/exp_generalized_comparison.yaml}"
echo "Config: $CONFIG"
echo "Start time: $(date)"

# Activate environment
source ~/miniconda3/bin/activate
echo ""
echo "Python: $(which python)"
python --version

# Run comparison
python scripts/run_generalized_comparison.py --config "$CONFIG"

echo ""
echo "End time: $(date)"
echo "=============================================="

