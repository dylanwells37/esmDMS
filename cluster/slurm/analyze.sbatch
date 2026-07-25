#!/usr/bin/env bash
# One CPU analyze job for a single dataset: enrichment / functional / regular
# popDMS (elbow gamma) / raw-LLR baselines plus the scale-matched alpha x gamma
# LLR-prior sweep for all three model sizes. Resources overridden per dataset.
#SBATCH --job-name=esmdms-analyze
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --cluster=htc
#SBATCH --partition=htc
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT}"
: "${VENV:?Set VENV}"
: "${CONFIG:?Set CONFIG to the per-dataset analyze JSON}"

cd "${REPO_ROOT}"
source "${VENV}/bin/activate"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${MPLCONFIGDIR}"

echo "job=${SLURM_JOB_ID:-local} host=$(hostname) config=${CONFIG}"
esmdms analyze "${CONFIG}"
echo "analyze complete: ${CONFIG}"
