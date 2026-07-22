#!/usr/bin/env bash
#SBATCH --job-name=esmdms-merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the repository checkout}"
: "${VENV:?Set VENV to the Python virtual environment}"
: "${DATASET:?Set DATASET to a canonical dataset directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the embedding/LLR output root}"
: "${NUM_SHARDS:?Set NUM_SHARDS to the completed array size}"

cd "${REPO_ROOT}"
source "${VENV}/bin/activate"

printf -v total_tag "%03d" "${NUM_SHARDS}"
printf -v first_shard "shard_%03d_of_%03d" 0 "${NUM_SHARDS}"
shard_root="${OUTPUT_ROOT}/shards"
merged_root="${OUTPUT_ROOT}/merged"
first_embedding_root="${shard_root}/${first_shard}/embeddings"
mkdir -p "${merged_root}/embeddings"

shopt -s nullglob
layer_files=("${first_embedding_root}"/layer_*.npz)
if (( ${#layer_files[@]} == 0 )); then
    echo "No layer artifacts found under ${first_embedding_root}" >&2
    exit 2
fi

for layer_file in "${layer_files[@]}"; do
    layer_name="$(basename "${layer_file}")"
    parts=("${shard_root}"/shard_*_of_"${total_tag}"/embeddings/"${layer_name}")
    if (( ${#parts[@]} != NUM_SHARDS )); then
        echo "${layer_name}: expected ${NUM_SHARDS} shards, found ${#parts[@]}" >&2
        exit 2
    fi
    esmdms merge "${DATASET}" "${parts[@]}" \
        --output "${merged_root}/embeddings/${layer_name}"
done

llr_parts=("${shard_root}"/shard_*_of_"${total_tag}"/llr.npz)
if (( ${#llr_parts[@]} != NUM_SHARDS )); then
    echo "LLR: expected ${NUM_SHARDS} shards, found ${#llr_parts[@]}" >&2
    exit 2
fi
esmdms merge "${DATASET}" "${llr_parts[@]}" \
    --output "${merged_root}/llr.npz"

echo "Merged artifacts written to ${merged_root}"
