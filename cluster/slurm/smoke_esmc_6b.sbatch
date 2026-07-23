#!/usr/bin/env bash
#SBATCH --job-name=esmc6b-smoke
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the repository checkout}"
: "${VENV:?Set VENV to the Python virtual environment}"
: "${OUTPUT:?Set OUTPUT to a persistent output directory}"

cd "${REPO_ROOT}"
source "${VENV}/bin/activate"

export HF_HOME="${HF_HOME:-${SCRATCH:-${HOME}/.cache}/huggingface}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}" "${OUTPUT}"

echo "job_id=${SLURM_JOB_ID:-local} host=$(hostname)"
nvidia-smi

gpu_log="${OUTPUT}/gpu_memory.csv"
nvidia-smi \
    --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu,power.draw \
    --format=csv,nounits \
    --loop=1 > "${gpu_log}" &
nvidia_smi_pid=$!
cleanup() {
    kill "${nvidia_smi_pid}" 2>/dev/null || true
    wait "${nvidia_smi_pid}" 2>/dev/null || true
}
trap cleanup EXIT

time_file="${OUTPUT}/process_resources.txt"
time_command="$(type -P time)"
if [[ -z "${time_command}" ]]; then
    echo "GNU time is required to measure peak host memory" >&2
    exit 2
fi
"${time_command}" -v -o "${time_file}" \
    python cluster/slurm/smoke_esmc_brca1.py \
    --model "${MODEL_ID:-biohub/ESMC-6B}" \
    --input "${BRCA1_INPUT:-imported_data/MV_BRCA1_Findlay_2018.csv}" \
    --output "${OUTPUT}" \
    --dtype "${DTYPE:-bf16}" \
    --min-gpu-memory-gib "${MIN_GPU_MEMORY_GIB:-48}"

cleanup
trap - EXIT
peak_gpu_memory_mib="$(awk -F, '
    NR > 1 {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5)
        if ($5 + 0 > peak) peak = $5 + 0
    }
    END { print peak + 0 }
' "${gpu_log}")"
echo "peak sampled whole-GPU memory=${peak_gpu_memory_mib} MiB"
echo "Process-level resource usage:"
cat "${time_file}"
echo "Slurm accounting (may be incomplete until the job exits):"
sstat --jobs="${SLURM_JOB_ID}.batch" \
    --format=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize -P 2>/dev/null || true
echo "Smoke test succeeded; results are in ${OUTPUT}"
