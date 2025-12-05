#!/bin/bash
#SBATCH --job-name=pinn_smoke
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# --- Quick smoke test for validation ----
: "${PROJECT_ROOT:=/users/tgillin/files/phys2630-final}"
: "${CONFIG_PATH:=${PROJECT_ROOT}/experiments/exp_hf_smoke.yaml}"
# ----------------------------------------

set -euo pipefail

mkdir -p "${PROJECT_ROOT}/logs"
echo "=============================================="
echo "PINN Smoke Test (HuggingFace Data)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Start time: $(date)"
echo ""

cd "${PROJECT_ROOT}"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate base 2>/dev/null || true
fi

echo "Python: $(which python)"
python --version
echo ""

# Use unbuffered Python output
export PYTHONUNBUFFERED=1

python scripts/run_comparison.py --config "${CONFIG_PATH}"

echo ""
echo "End time: $(date)"
echo "=============================================="

