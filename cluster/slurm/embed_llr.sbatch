#!/usr/bin/env bash
#SBATCH --job-name=esmdms-embed-llr
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the repository checkout}"
: "${VENV:?Set VENV to the Python virtual environment}"
: "${DATASET:?Set DATASET to a canonical dataset directory}"
: "${MODEL_ID:?Set MODEL_ID to a Hugging Face ESM-C model id}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a persistent output directory}"
: "${NUM_SHARDS:?Set NUM_SHARDS to the Slurm array size}"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_id < 0 || task_id >= NUM_SHARDS )); then
    echo "Array task ${task_id} is outside NUM_SHARDS=${NUM_SHARDS}" >&2
    exit 2
fi

device_mode="${DEVICE_MODE:-gpu}"
case "${device_mode}" in
    gpu)
        torch_device="cuda"
        model_dtype="${DTYPE:-bf16}"
        ;;
    cpu)
        torch_device="cpu"
        model_dtype="${DTYPE:-fp32}"
        ;;
    *)
        echo "DEVICE_MODE must be gpu or cpu, received ${device_mode}" >&2
        exit 2
        ;;
esac

cd "${REPO_ROOT}"
source "${VENV}/bin/activate"

export HF_HOME="${HF_HOME:-${SCRATCH:-${HOME}/.cache}/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${HF_HOME}" "${MPLCONFIGDIR}"

printf -v shard_tag "shard_%03d_of_%03d" "${task_id}" "${NUM_SHARDS}"
shard_output="${OUTPUT_ROOT}/shards/${shard_tag}"

if [[ -n "${LAYERS:-}" ]]; then
    read -r -a selected_layers <<< "${LAYERS}"
else
    selected_layers=(all)
fi

echo "dataset=${DATASET}"
echo "model=${MODEL_ID} device=${torch_device} dtype=${model_dtype}"
echo "shard=${task_id}/${NUM_SHARDS} output=${shard_output}"

arguments=(
    "${DATASET}"
    --model "${MODEL_ID}"
    --layers "${selected_layers[@]}"
    --pooling "${POOLING:-max}"
    --window-size "${WINDOW_SIZE:-2048}"
    --device "${torch_device}"
    --dtype "${model_dtype}"
    --shard-index "${task_id}"
    --num-shards "${NUM_SHARDS}"
    --output "${shard_output}"
)

esmdms embed-llr "${arguments[@]}"
