#!/usr/bin/env bash
# One-command launcher for the model-size LLR-prior benchmark.
#
#   bash experiments/model_size_llr/submit_experiment.sh
#
# Generates the per-dataset analyze configs and job manifests, then submits the
# controller job (on htc) that runs the whole pipeline hands-off:
#   process datasets -> 15 GPU LLR jobs -> 5 CPU analyze jobs -> report.
# Submit it and walk away; results land under results/model_size_llr/aggregate.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${script_dir}/../.." && pwd)}"
VENV="${VENV:-${REPO_ROOT}/.venv}"

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
    echo "REPO_ROOT ${REPO_ROOT} is not the repository root" >&2
    exit 2
fi
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "Python environment not found at ${VENV}; run 'make setup' or set VENV" >&2
    exit 2
fi

results_root="${REPO_ROOT}/results/model_size_llr"
config_dir="${results_root}/configs"
aggregate_dir="${results_root}/aggregate"
controller="${REPO_ROOT}/cluster/slurm/experiment_controller.sbatch"

mkdir -p "${results_root}" "${config_dir}/logs" "${aggregate_dir}" "${REPO_ROOT}/artifacts"

# Generate configs + manifests (fast, login node).
source "${VENV}/bin/activate"
python "${script_dir}/make_configs.py" \
    --repo-root "${REPO_ROOT}" \
    --config-dir "${config_dir}" \
    --results-root "${results_root}"

# A fresh run should not inherit a stale GPU-job state file.
rm -f "${config_dir}/llr_state.tsv"

controller_id="$(sbatch --parsable -M htc \
    --output="${config_dir}/logs/controller-%j.out" \
    --error="${config_dir}/logs/controller-%j.err" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",VENV="${VENV}",CONFIG_DIR="${config_dir}",RESULTS_ROOT="${results_root}",AGGREGATE_DIR="${aggregate_dir}",CONTROLLER_SCRIPT="${controller}" \
    "${controller}")"
controller_id="${controller_id%%;*}"

cat <<EOF
Submitted controller job ${controller_id} on htc.
It will: process datasets -> submit 15 GPU LLR jobs -> poll -> submit 5 analyze
jobs -> submit report. Monitor with:
  squeue -M htc -u \$USER
  squeue -M gpu -u \$USER
  tail -f ${config_dir}/logs/controller-${controller_id}.out
Final outputs: ${aggregate_dir}/ (baselines.csv, prior_sweeps.csv, summary.csv,
multi_dataset_llr_prior.html)
EOF
