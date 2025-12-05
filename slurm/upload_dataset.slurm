#!/bin/bash
#SBATCH --job-name=hf_upload
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# --- Upload FBM trajectories to Hugging Face ---
# Set HF_TOKEN before running:
#   export HF_TOKEN=<your_token>
#   sbatch slurm/upload_dataset.slurm
# ------------------------------------------------

: "${PROJECT_ROOT:=/users/tgillin/files/phys2630-final}"

set -euo pipefail

mkdir -p "${PROJECT_ROOT}/logs"
echo "=============================================="
echo "Upload FBM Trajectories to Hugging Face"
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

# Check for HF token
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN environment variable not set!"
    echo "Set it with: export HF_TOKEN=<your_token>"
    exit 1
fi

echo "Python: $(which python)"
python --version
echo ""

# Install datasets if needed
pip install -q datasets huggingface_hub

python scripts/upload_dataset.py --token "${HF_TOKEN}"

echo ""
echo "End time: $(date)"
echo "=============================================="


