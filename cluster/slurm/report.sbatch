#!/usr/bin/env bash
# Aggregate the five per-dataset analyze outputs and execute the reporting
# notebook to HTML. Submitted with an afterok dependency on the analyze jobs
# (same cluster, so the dependency is honored).
#SBATCH --job-name=esmdms-report
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --partition=htc
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT}"
: "${VENV:?Set VENV}"
: "${RESULTS_ROOT:?Set RESULTS_ROOT (per-dataset analyze output root)}"
: "${CONFIG_DIR:?Set CONFIG_DIR (holds datasets.txt)}"
: "${AGGREGATE_DIR:?Set AGGREGATE_DIR}"

cd "${REPO_ROOT}"
source "${VENV}/bin/activate"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}}"
mkdir -p "${MPLCONFIGDIR}" "${AGGREGATE_DIR}"

echo "job=${SLURM_JOB_ID:-local} host=$(hostname)"
python experiments/model_size_llr/aggregate.py \
    --results-root "${RESULTS_ROOT}" \
    --datasets-file "${CONFIG_DIR}/datasets.txt" \
    --output-dir "${AGGREGATE_DIR}" \
    --primary-cutoff "${PRIMARY_CUTOFF:-0}"

# Execute the reporting notebook against the aggregate results into AGGREGATE_DIR,
# leaving the committed source notebook unexecuted. The notebook reads the results
# location from ESMDMS_AGGREGATE_DIR.
export ESMDMS_AGGREGATE_DIR="${AGGREGATE_DIR}"
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=1800 \
    --output "${AGGREGATE_DIR}/multi_dataset_llr_prior.ipynb" \
    multi_dataset_llr_prior.ipynb
jupyter nbconvert --to html \
    --output-dir "${AGGREGATE_DIR}" \
    "${AGGREGATE_DIR}/multi_dataset_llr_prior.ipynb"

echo "report complete: ${AGGREGATE_DIR}"
