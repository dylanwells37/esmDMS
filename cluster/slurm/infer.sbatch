#!/usr/bin/env bash
#SBATCH --job-name=esmdms-infer
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the repository checkout}"
: "${VENV:?Set VENV to the Python virtual environment}"
: "${DATASET:?Set DATASET to a canonical dataset directory}"
: "${OUTPUT:?Set OUTPUT to the fitness .npz path}"

if [[ -n "${FEATURES:-}" && -n "${PRIOR:-}" ]]; then
    echo "Do not combine FEATURES and PRIOR: LLR priors use substitution features." >&2
    exit 2
fi

cd "${REPO_ROOT}"
source "${VENV}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

arguments=(
    "${DATASET}"
    --gamma "${GAMMA:-0.1}"
    --output "${OUTPUT}"
)
if [[ -n "${FEATURES:-}" ]]; then
    arguments+=(--features "${FEATURES}")
fi
if [[ -n "${PRIOR:-}" ]]; then
    arguments+=(--prior "${PRIOR}" --alpha "${ALPHA:-1.0}")
fi

esmdms infer "${arguments[@]}"
