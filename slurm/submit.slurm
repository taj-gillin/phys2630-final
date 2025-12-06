#!/bin/bash
#SBATCH --job-name=andi
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# =============================================================================
# SLURM Submission Script for Anomalous Diffusion Project
# =============================================================================
#
# Usage:
#   sbatch slurm/submit.slurm train configs/models/lstm.yaml
#   sbatch slurm/submit.slurm compare configs/compare.yaml
#   sbatch slurm/submit.slurm generate configs/data/generate.yaml
#
# Job Types:
#   train     - Train a model (requires model config)
#   compare   - Run method comparison
#   generate  - Generate and upload dataset to HF
#
# =============================================================================

# Project root - use SLURM_SUBMIT_DIR (directory from which sbatch was called)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

# Change to project directory FIRST
cd "${PROJECT_ROOT}" || { echo "Error: Cannot cd to ${PROJECT_ROOT}"; exit 1; }

# Now parse arguments
JOB_TYPE="${1:-train}"
CONFIG_PATH="${2:-}"

# Map job type to script
case "${JOB_TYPE}" in
    train)
        SCRIPT="scripts/train.py"
        ;;
    compare)
        SCRIPT="scripts/run_comparison.py"
        ;;
    generate)
        SCRIPT="scripts/generate_data.py"
        CONFIG_PATH="${CONFIG_PATH:-configs/data/generate.yaml}"
        ;;
    *)
        echo "Error: Unknown job type '${JOB_TYPE}'"
        echo "Valid types: train, compare, generate"
        exit 1
        ;;
esac

# Validate config exists
if [ -n "${CONFIG_PATH}" ] && [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: Config not found: ${CONFIG_PATH}"
    echo "Working directory: $(pwd)"
    echo "Available configs:"
    ls configs/models/ 2>/dev/null || echo "  (none)"
    exit 1
fi

# Create logs directory
mkdir -p "${PROJECT_ROOT}/logs"

# Header
echo "=============================================================================="
echo "ANDI: Anomalous Diffusion Inference"
echo "=============================================================================="
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Job Type:   ${JOB_TYPE}"
echo "Config:     ${CONFIG_PATH:-none}"
echo "Node:       ${SLURM_NODELIST:-localhost}"
echo "PWD:        $(pwd)"
echo "Start:      $(date)"
echo ""

# Environment setup (don't fail if conda not available)
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Python:     $(which python)"
python -c "
import torch
print(f'PyTorch:    {torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA:       {torch.version.cuda}')
    print(f'GPU:        {torch.cuda.get_device_name(0)}')
else:
    print('CUDA:       Not available')
"
echo ""

# Run
echo "=============================================================================="
echo "python ${SCRIPT} --config ${CONFIG_PATH}"
echo "=============================================================================="
echo ""

python "${SCRIPT}" --config "${CONFIG_PATH}"

echo ""
echo "=============================================================================="
echo "Completed:  $(date)"
echo "=============================================================================="
