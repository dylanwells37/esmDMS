#!/usr/bin/env bash
# Submit a sharded embedding/LLR, merge, SAE, and inference dependency graph.

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the repository checkout}"
: "${VENV:?Set VENV to the Python virtual environment}"
: "${DATASET:?Set DATASET to a canonical dataset directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a persistent output directory}"

model_size="${MODEL_SIZE:-300m}"
case "${model_size}" in
    300m)
        model_id="${MODEL_ID:-biohub/ESMC-300M}"
        final_layer="${SAE_LAYER:-30}"
        embed_mem="${EMBED_MEM:-32G}"
        embed_time="${EMBED_TIME:-12:00:00}"
        sae_features="${SAE_FEATURES:-1920}"
        ;;
    600m)
        model_id="${MODEL_ID:-biohub/ESMC-600M}"
        final_layer="${SAE_LAYER:-36}"
        embed_mem="${EMBED_MEM:-48G}"
        embed_time="${EMBED_TIME:-18:00:00}"
        sae_features="${SAE_FEATURES:-2304}"
        ;;
    3b)
        : "${MODEL_ID:?MODEL_SIZE=3b requires a compatible custom ESM-C MODEL_ID}"
        model_id="${MODEL_ID}"
        final_layer="${SAE_LAYER:-36}"
        embed_mem="${EMBED_MEM:-96G}"
        embed_time="${EMBED_TIME:-36:00:00}"
        sae_features="${SAE_FEATURES:-5120}"
        ;;
    6b)
        model_id="${MODEL_ID:-biohub/ESMC-6B}"
        final_layer="${SAE_LAYER:-80}"
        embed_mem="${EMBED_MEM:-160G}"
        embed_time="${EMBED_TIME:-48:00:00}"
        sae_features="${SAE_FEATURES:-5120}"
        ;;
    *)
        echo "MODEL_SIZE must be 300m, 600m, 3b, or 6b" >&2
        exit 2
        ;;
esac

num_shards="${NUM_SHARDS:-4}"
if (( num_shards <= 0 )); then
    echo "NUM_SHARDS must be positive" >&2
    exit 2
fi
device_mode="${DEVICE_MODE:-gpu}"
script_root="${REPO_ROOT}/cluster/slurm"
mkdir -p "${OUTPUT_ROOT}/results" "${OUTPUT_ROOT}/sae"

gpu_request=""
if [[ "${device_mode}" == gpu ]]; then
    gpu_request="--gres=${GPU_RESOURCE:-gpu:1}"
elif [[ "${device_mode}" != cpu ]]; then
    echo "DEVICE_MODE must be gpu or cpu" >&2
    exit 2
fi

embed_export="ALL,REPO_ROOT=${REPO_ROOT},VENV=${VENV}"
embed_export+=",DATASET=${DATASET},MODEL_ID=${model_id}"
embed_export+=",OUTPUT_ROOT=${OUTPUT_ROOT},NUM_SHARDS=${num_shards}"
embed_export+=",DEVICE_MODE=${device_mode},DTYPE=${DTYPE:-}"
embed_job="$(sbatch --parsable \
    --array="0-$((num_shards - 1))" \
    --mem="${embed_mem}" \
    --time="${embed_time}" \
    ${gpu_request:+"${gpu_request}"} \
    --export="${embed_export}" \
    "${script_root}/embed_llr.sbatch")"
embed_job="${embed_job%%;*}"

merge_export="ALL,REPO_ROOT=${REPO_ROOT},VENV=${VENV},DATASET=${DATASET}"
merge_export+=",OUTPUT_ROOT=${OUTPUT_ROOT},NUM_SHARDS=${num_shards}"
merge_job="$(sbatch --parsable \
    --dependency="afterok:${embed_job}" \
    --export="${merge_export}" \
    "${script_root}/merge_embed_llr.sbatch")"
merge_job="${merge_job%%;*}"

embedding="${OUTPUT_ROOT}/merged/embeddings/layer_${final_layer}.npz"
sae_artifact="${OUTPUT_ROOT}/sae/layer_${final_layer}.npz"
sae_model="${OUTPUT_ROOT}/sae/layer_${final_layer}.pt"
sae_export="ALL,REPO_ROOT=${REPO_ROOT},VENV=${VENV},DATASET=${DATASET}"
sae_export+=",EMBEDDING=${embedding},OUTPUT_ARTIFACT=${sae_artifact}"
sae_export+=",OUTPUT_MODEL=${sae_model},DEVICE_MODE=${device_mode}"
sae_export+=",SAE_FEATURES=${sae_features},SAE_MODE=${SAE_MODE:-batchtopk}"
sae_export+=",K=${K:-64},EPOCHS=${EPOCHS:-200}"
sae_job="$(sbatch --parsable \
    --dependency="afterok:${merge_job}" \
    ${gpu_request:+"${gpu_request}"} \
    --export="${sae_export}" \
    "${script_root}/sae.sbatch")"
sae_job="${sae_job%%;*}"

embedding_infer_export="ALL,REPO_ROOT=${REPO_ROOT},VENV=${VENV}"
embedding_infer_export+=",DATASET=${DATASET},FEATURES=${embedding}"
embedding_infer_export+=",GAMMA=${GAMMA:-0.1}"
embedding_infer_export+=",OUTPUT=${OUTPUT_ROOT}/results/embedding_fitness.npz"
embedding_infer_job="$(sbatch --parsable \
    --dependency="afterok:${merge_job}" \
    --export="${embedding_infer_export}" \
    "${script_root}/infer.sbatch")"
embedding_infer_job="${embedding_infer_job%%;*}"

prior_infer_export="ALL,REPO_ROOT=${REPO_ROOT},VENV=${VENV},DATASET=${DATASET}"
prior_infer_export+=",PRIOR=${OUTPUT_ROOT}/merged/llr.npz"
prior_infer_export+=",GAMMA=${GAMMA:-0.1},ALPHA=${ALPHA:-1.0}"
prior_infer_export+=",OUTPUT=${OUTPUT_ROOT}/results/llr_prior_fitness.npz"
prior_infer_job="$(sbatch --parsable \
    --dependency="afterok:${merge_job}" \
    --export="${prior_infer_export}" \
    "${script_root}/infer.sbatch")"
prior_infer_job="${prior_infer_job%%;*}"

sae_infer_export="ALL,REPO_ROOT=${REPO_ROOT},VENV=${VENV}"
sae_infer_export+=",DATASET=${DATASET},FEATURES=${sae_artifact}"
sae_infer_export+=",GAMMA=${GAMMA:-0.1}"
sae_infer_export+=",OUTPUT=${OUTPUT_ROOT}/results/sae_fitness.npz"
sae_infer_job="$(sbatch --parsable \
    --dependency="afterok:${sae_job}" \
    --export="${sae_infer_export}" \
    "${script_root}/infer.sbatch")"
sae_infer_job="${sae_infer_job%%;*}"

printf '%-24s %s\n' \
    "embed_llr_array" "${embed_job}" \
    "merge" "${merge_job}" \
    "sae" "${sae_job}" \
    "embedding_inference" "${embedding_infer_job}" \
    "llr_prior_inference" "${prior_infer_job}" \
    "sae_inference" "${sae_infer_job}"
