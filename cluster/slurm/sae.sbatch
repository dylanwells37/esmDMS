#!/usr/bin/env bash
#SBATCH --job-name=esmdms-sae
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the repository checkout}"
: "${VENV:?Set VENV to the Python virtual environment}"
: "${DATASET:?Set DATASET to a canonical dataset directory}"
: "${EMBEDDING:?Set EMBEDDING to one merged layer artifact}"
: "${OUTPUT_ARTIFACT:?Set OUTPUT_ARTIFACT to the SAE .npz path}"
: "${OUTPUT_MODEL:?Set OUTPUT_MODEL to the SAE .pt path}"

device_mode="${DEVICE_MODE:-gpu}"
case "${device_mode}" in
    gpu) torch_device="cuda" ;;
    cpu) torch_device="cpu" ;;
    *)
        echo "DEVICE_MODE must be gpu or cpu, received ${device_mode}" >&2
        exit 2
        ;;
esac

cd "${REPO_ROOT}"
source "${VENV}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mode="${SAE_MODE:-batchtopk}"
arguments=(
    "${DATASET}"
    "${EMBEDDING}"
    --mode "${mode}"
    --epochs "${EPOCHS:-200}"
    --batch-size "${BATCH_SIZE:-64}"
    --learning-rate "${LEARNING_RATE:-0.001}"
    --sparsity "${SPARSITY:-0.001}"
    --seed "${SEED:-42}"
    --device "${torch_device}"
    --model-output "${OUTPUT_MODEL}"
    --output "${OUTPUT_ARTIFACT}"
)
if [[ -n "${SAE_FEATURES:-}" ]]; then
    arguments+=(--features "${SAE_FEATURES}")
fi
if [[ "${mode}" != normal ]]; then
    arguments+=(--k "${K:-64}")
fi

esmdms sae "${arguments[@]}"
