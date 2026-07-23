#!/usr/bin/env bash
# Submit the one-sequence ESM-C 6B GPU smoke test.

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the repository checkout}"
: "${VENV:?Set VENV to the Python virtual environment}"
: "${OUTPUT:?Set OUTPUT to a persistent output directory}"

mkdir -p "${OUTPUT}"

arguments=(
    --parsable
    --mem="${MEMORY:-160G}"
    --time="${TIME_LIMIT:-02:00:00}"
    --gres="${GPU_RESOURCE:-gpu:1}"
    --output="${OUTPUT}/slurm-%x-%j.out"
    --error="${OUTPUT}/slurm-%x-%j.err"
)
if [[ -n "${GPU_CONSTRAINT:-}" ]]; then
    arguments+=(--constraint="${GPU_CONSTRAINT}")
fi
if [[ -n "${PARTITION:-}" ]]; then
    arguments+=(--partition="${PARTITION}")
fi
if [[ -n "${ACCOUNT:-}" ]]; then
    arguments+=(--account="${ACCOUNT}")
fi
if [[ -n "${QOS:-}" ]]; then
    arguments+=(--qos="${QOS}")
fi

exports="ALL,REPO_ROOT=${REPO_ROOT},VENV=${VENV},OUTPUT=${OUTPUT}"
exports+=",MODEL_ID=${MODEL_ID:-biohub/ESMC-6B},DTYPE=${DTYPE:-bf16}"
exports+=",MIN_GPU_MEMORY_GIB=${MIN_GPU_MEMORY_GIB:-48}"

job_id="$(sbatch "${arguments[@]}" --export="${exports}" \
    "${REPO_ROOT}/cluster/slurm/smoke_esmc_6b.sbatch")"
job_id="${job_id%%;*}"
echo "Submitted ESM-C 6B smoke test job ${job_id}"
echo "Follow it with: tail -f ${OUTPUT}/slurm-esmc6b-smoke-${job_id}.out"
echo "After completion: sacct -j ${job_id} --format=JobID,State,Elapsed,MaxRSS,ReqMem"
