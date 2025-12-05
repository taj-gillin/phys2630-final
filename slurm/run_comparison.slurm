#!/bin/bash
#SBATCH --job-name=pinn_comparison
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# --- user knobs (override with env vars via sbatch --export) ----
: "${PROJECT_ROOT:=/users/tgillin/files/phys2630-final}"
: "${CONFIG_PATH:=${PROJECT_ROOT}/experiments/exp_hf_comparison.yaml}"
# ----------------------------------------------------------------

set -euo pipefail

mkdir -p "${PROJECT_ROOT}/logs"
echo "=============================================="
echo "PINN Method Comparison Experiment"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "Project root: ${PROJECT_ROOT}"
echo "Config: ${CONFIG_PATH}"
echo "Start time: $(date)"
echo ""

cd "${PROJECT_ROOT}"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate base 2>/dev/null || true
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "Python: $(which python)"
python --version
echo ""

# Use unbuffered Python output
export PYTHONUNBUFFERED=1

python scripts/run_comparison.py --config "${CONFIG_PATH}"

echo ""
echo "End time: $(date)"
echo "=============================================="

