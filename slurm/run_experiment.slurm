#!/bin/bash
#SBATCH --job-name=pinn_anom_diff
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# --- user knobs (override with env vars via sbatch --export or exports) ----
: "${PROJECT_ROOT:=/users/tgillin/files/phys2630-final}"
: "${ENV_PATH:=${PROJECT_ROOT}/.venv}"        # change if using a module-based env
: "${CONFIG_PATH:=${PROJECT_ROOT}/experiments/exp_single_alpha.yaml}"
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p "${PROJECT_ROOT}/logs"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node list: ${SLURM_NODELIST:-N/A}"
echo "Project root: ${PROJECT_ROOT}"
echo "Config: ${CONFIG_PATH}"
echo "Start time: $(date)"

cd "${PROJECT_ROOT}"

if [ -d "${ENV_PATH}" ]; then
  source "${ENV_PATH}/bin/activate"
else
  echo "Warning: ${ENV_PATH} not found; using system Python."
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Python: $(which python)"
python --version

python scripts/run_experiment.py --config "${CONFIG_PATH}"

echo "End time: $(date)"

