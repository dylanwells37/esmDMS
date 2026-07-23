#!/usr/bin/env bash
# Master controller for the model-size LLR-prior experiment.
#
# The gpu and htc clusters are not federated, so a cross-cluster afterok
# dependency is impossible. Instead this lightweight htc job orchestrates the
# whole run: it processes the datasets, submits the GPU LLR jobs, polls them to
# completion, then submits the same-cluster analyze -> report DAG and exits. If
# it approaches its walltime while GPU jobs are still queued it resubmits itself
# and resumes from the recorded job-id state file.
#SBATCH --job-name=esmdms-controller
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --cluster=htc
#SBATCH --partition=htc
#SBATCH --mem=8G
#SBATCH --time=08:00:00

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT}"
: "${VENV:?Set VENV}"
: "${CONFIG_DIR:?Set CONFIG_DIR}"
: "${RESULTS_ROOT:?Set RESULTS_ROOT}"
: "${AGGREGATE_DIR:?Set AGGREGATE_DIR}"
: "${CONTROLLER_SCRIPT:?Set CONTROLLER_SCRIPT to the path of this script}"

script_root="${REPO_ROOT}/cluster/slurm"
state_file="${CONFIG_DIR}/llr_state.tsv"
poll_seconds="${POLL_SECONDS:-60}"
max_wait_seconds="${MAX_WAIT_SECONDS:-25200}"  # resubmit before the 8h walltime
export HF_HOME="${HF_HOME:-${SCRATCH:-${HOME}/.cache}/huggingface}"

cd "${REPO_ROOT}"
source "${VENV}/bin/activate"
mkdir -p "${HF_HOME}" "${RESULTS_ROOT}" "${CONFIG_DIR}/logs" "${REPO_ROOT}/artifacts"

log() { echo "[controller $(date +%H:%M:%S)] $*"; }

# strip the "jobid;cluster" suffix that --parsable emits on a multi-cluster site
bare_id() { printf '%s' "${1%%;*}"; }

log "job=${SLURM_JOB_ID:-local} host=$(hostname)"

# --- 1. Process any missing canonical datasets (CPU, fast, no network). --------
mapfile -t datasets < "${CONFIG_DIR}/datasets.txt"
missing=()
for dataset in "${datasets[@]}"; do
    [[ -f "${REPO_ROOT}/datasets/${dataset}/dataset.json" ]] || missing+=("${dataset}")
done
if (( ${#missing[@]} > 0 )); then
    log "processing datasets: ${missing[*]}"
    esmdms process imported_data --output "${REPO_ROOT}/datasets" --datasets "${missing[@]}"
else
    log "all datasets already processed"
fi

# --- 2. Submit the 15 GPU LLR jobs once; record ids in the state file. ---------
if [[ ! -f "${state_file}" ]]; then
    log "submitting GPU LLR jobs"
    : > "${state_file}"
    while IFS=$'\t' read -r dataset size model_id output mem time_limit; do
        [[ -z "${dataset}" ]] && continue
        if [[ -f "${output}" ]]; then
            log "skip ${dataset}/${size}: artifact exists"
            printf '%s\t%s\t%s\t%s\n' "${dataset}" "${size}" "SKIP" "${output}" >> "${state_file}"
            continue
        fi
        jid="$(bare_id "$(sbatch --parsable -M gpu \
            --mem="${mem}" --time="${time_limit}" \
            --output="${CONFIG_DIR}/logs/llr-${dataset}-${size}-%j.out" \
            --error="${CONFIG_DIR}/logs/llr-${dataset}-${size}-%j.err" \
            --export=ALL,REPO_ROOT="${REPO_ROOT}",VENV="${VENV}",HF_HOME="${HF_HOME}",DATASET_DIR="${REPO_ROOT}/datasets/${dataset}",MODEL_ID="${model_id}",OUTPUT="${output}" \
            "${script_root}/llr.sbatch")")"
        log "submitted ${dataset}/${size} -> gpu job ${jid}"
        printf '%s\t%s\t%s\t%s\n' "${dataset}" "${size}" "${jid}" "${output}" >> "${state_file}"
    done < "${CONFIG_DIR}/llr_jobs.tsv"
else
    log "resuming from existing state file ${state_file}"
fi

# --- 3. Poll GPU jobs to completion; self-resubmit near the walltime. ----------
gpu_ids="$(awk -F'\t' '$3 ~ /^[0-9]+$/ {print $3}' "${state_file}" | paste -sd, -)"
if [[ -n "${gpu_ids}" ]]; then
    while :; do
        active="$(squeue -M gpu -h -j "${gpu_ids}" -t pending,running -o '%A' 2>/dev/null | grep -c '[0-9]' || true)"
        (( active == 0 )) && { log "all GPU jobs left the queue"; break; }
        if (( SECONDS >= max_wait_seconds )); then
            log "near walltime with ${active} GPU jobs left; resubmitting controller"
            sbatch -M htc --export=ALL "${CONTROLLER_SCRIPT}"
            exit 0
        fi
        log "waiting on ${active} GPU job(s); sleeping ${poll_seconds}s"
        sleep "${poll_seconds}"
    done
fi

# --- 4. Determine which datasets have all three LLR artifacts present. ---------
complete=()
for dataset in "${datasets[@]}"; do
    ok=1
    for size in 300M 600M 6B; do
        [[ -f "${REPO_ROOT}/artifacts/${dataset}/${size}_llr.npz" ]] || ok=0
    done
    if (( ok == 1 )); then
        complete+=("${dataset}")
    else
        log "WARNING: ${dataset} is missing one or more LLR artifacts; skipping its analyze job"
    fi
done
if (( ${#complete[@]} == 0 )); then
    log "no dataset has a complete LLR set; aborting before analyze"
    exit 1
fi

# --- 5. Submit one analyze job per complete dataset (htc, same cluster). -------
declare -A analyze_ids=()
while IFS=$'\t' read -r dataset config mem time_limit; do
    [[ -z "${dataset}" ]] && continue
    printf '%s\n' "${complete[@]}" | grep -qx "${dataset}" || continue
    jid="$(bare_id "$(sbatch --parsable -M htc \
        --mem="${mem}" --time="${time_limit}" \
        --output="${CONFIG_DIR}/logs/analyze-${dataset}-%j.out" \
        --error="${CONFIG_DIR}/logs/analyze-${dataset}-%j.err" \
        --export=ALL,REPO_ROOT="${REPO_ROOT}",VENV="${VENV}",CONFIG="${config}" \
        "${script_root}/analyze.sbatch")")"
    analyze_ids["${dataset}"]="${jid}"
    log "submitted analyze ${dataset} -> htc job ${jid}"
done < "${CONFIG_DIR}/analyze_jobs.tsv"

# --- 6. Submit the report job, afterok on every analyze job. -------------------
dep="$(IFS=:; echo "${analyze_ids[*]}")"
report_id="$(bare_id "$(sbatch --parsable -M htc \
    --dependency="afterok:${dep}" \
    --output="${CONFIG_DIR}/logs/report-%j.out" \
    --error="${CONFIG_DIR}/logs/report-%j.err" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",VENV="${VENV}",RESULTS_ROOT="${RESULTS_ROOT}",CONFIG_DIR="${CONFIG_DIR}",AGGREGATE_DIR="${AGGREGATE_DIR}" \
    "${script_root}/report.sbatch")")"
log "submitted report -> htc job ${report_id} (afterok:${dep})"
log "controller done; analyze + report will finish on their own"
