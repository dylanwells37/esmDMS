#!/bin/bash
set -euo pipefail

# Submit ESM embedding jobs for the five datasets used in the walkthrough:
#   TpoR, Ube4b, BRCA1, BF520, BG505
#
# Each dataset is submitted as a 10-task Slurm array using
# embedding_scripts/submit_embed_array.sh. A dependent merge job is submitted
# after each array completes successfully.
#
# Usage:
#   bash job_scripts/submit_five_dataset_embeddings.sh [embedding_method] [n_chunks] [extra embed args...]
#
# Examples:
#   bash job_scripts/submit_five_dataset_embeddings.sh
#   bash job_scripts/submit_five_dataset_embeddings.sh cls
#   bash job_scripts/submit_five_dataset_embeddings.sh mutation_site
#   bash job_scripts/submit_five_dataset_embeddings.sh mutation_site 10 --pool_mutations
#   DRY_RUN=1 bash job_scripts/submit_five_dataset_embeddings.sh cls

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
N_CHUNKS="${2:-10}"
EMBEDDING_METHOD="${1:-mean_pool}"

if [[ "$EMBEDDING_METHOD" != "mean_pool" && "$EMBEDDING_METHOD" != "cls" && "$EMBEDDING_METHOD" != "mutation_site" ]]; then
    echo "ERROR: embedding_method must be one of: mean_pool, cls, mutation_site" >&2
    exit 1
fi

if ! [[ "$N_CHUNKS" =~ ^[0-9]+$ ]] || [[ "$N_CHUNKS" -lt 1 ]]; then
    echo "ERROR: n_chunks must be a positive integer" >&2
    exit 1
fi

shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))
EXTRA_ARGS=("$@")

DATASETS=(TpoR Ube4b BRCA1 BF520 BG505)

method_suffix="$EMBEDDING_METHOD"
for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
    if [[ "$arg" == "--pool_mutations" && "$EMBEDDING_METHOD" == "mutation_site" ]]; then
        method_suffix="mutation_site_pooled"
    fi
done

output_file_for_dataset() {
    local dataset="$1"
    if [[ "$method_suffix" == "mean_pool" ]]; then
        echo "${dataset}_embeddings.pkl"
    else
        echo "${dataset}_${method_suffix}_embeddings.pkl"
    fi
}

run_cmd() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf 'DRY_RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

cd "$PROJECT_DIR"
mkdir -p job_outs

echo "Project dir      : $PROJECT_DIR"
echo "Datasets         : ${DATASETS[*]}"
echo "Workers/dataset  : $N_CHUNKS"
echo "Embedding method : $EMBEDDING_METHOD"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "Extra args       : ${EXTRA_ARGS[*]}"
else
    echo "Extra args       : <none>"
fi

for dataset in "${DATASETS[@]}"; do
    config="embedding_scripts/embed_config_${dataset}.json"
    output_file="$(output_file_for_dataset "$dataset")"

    if [[ ! -f "$config" ]]; then
        echo "ERROR: missing config $config" >&2
        exit 1
    fi

    echo
    echo "Submitting $dataset -> $output_file"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        run_cmd sbatch --array="0-$((N_CHUNKS - 1))" \
            embedding_scripts/submit_embed_array.sh \
            "$config" "$output_file" "$N_CHUNKS" \
            --embedding_method "$EMBEDDING_METHOD" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        run_cmd sbatch --dependency="afterok:<${dataset}_array_job_id>" \
            --job-name="merge_${dataset}_${method_suffix}" \
            --output="job_outs/merge_${dataset}_${method_suffix}-%j.out" \
            --error="job_outs/merge_${dataset}_${method_suffix}-%j.err" \
            --wrap="source ~/popDMS/esmDMS/.venv/bin/activate && cd $PROJECT_DIR && python embedding_scripts/merge_embedding_chunks.py $output_file $N_CHUNKS --delete_chunks"
        continue
    fi

    array_job_id="$(
        sbatch --parsable --array="0-$((N_CHUNKS - 1))" \
            embedding_scripts/submit_embed_array.sh \
            "$config" "$output_file" "$N_CHUNKS" \
            --embedding_method "$EMBEDDING_METHOD" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    )"
    echo "  array job: $array_job_id"

    merge_job_id="$(
        sbatch --parsable --dependency="afterok:${array_job_id}" \
            --job-name="merge_${dataset}_${method_suffix}" \
            --output="job_outs/merge_${dataset}_${method_suffix}-%j.out" \
            --error="job_outs/merge_${dataset}_${method_suffix}-%j.err" \
            --wrap="source ~/popDMS/esmDMS/.venv/bin/activate && cd $PROJECT_DIR && python embedding_scripts/merge_embedding_chunks.py $output_file $N_CHUNKS --delete_chunks"
    )"
    echo "  merge job: $merge_job_id"
done

echo
echo "Submitted embedding arrays and dependent merge jobs."
