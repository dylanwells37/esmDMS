#!/usr/bin/env bash
# One GPU masked-marginal LLR job for a single dataset x ESM-C model.
# Submitted by experiment_controller.sbatch (resources overridden per model).
#SBATCH --job-name=esmdms-llr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT}"
: "${VENV:?Set VENV}"
: "${DATASET_DIR:?Set DATASET_DIR to the canonical dataset directory}"
: "${MODEL_ID:?Set MODEL_ID to a Hugging Face ESM-C id}"
: "${OUTPUT:?Set OUTPUT to the llr .npz path}"

cd "${REPO_ROOT}"
source "${VENV}/bin/activate"

export HF_HOME="${HF_HOME:-${SCRATCH:-${HOME}/.cache}/huggingface}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}" "$(dirname "${OUTPUT}")"

echo "job=${SLURM_JOB_ID:-local} host=$(hostname) model=${MODEL_ID}"
echo "dataset=${DATASET_DIR}"
echo "output=${OUTPUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# One masked-marginal forward pass per unique mutation position. BRCA2's
# C-terminal window is read from dataset.json automatically.
esmdms llr "${DATASET_DIR}" \
    --model "${MODEL_ID}" \
    --window-size "${WINDOW_SIZE:-2048}" \
    --device cuda \
    --dtype "${DTYPE:-bf16}" \
    --output "${OUTPUT}"

echo "LLR complete: ${OUTPUT}"
