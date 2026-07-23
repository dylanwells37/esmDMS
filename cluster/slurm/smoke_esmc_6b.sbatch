#!/usr/bin/env bash
#SBATCH --job-name=a100-job
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --mem=48GB
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

set -euo pipefail

repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${repo_root}" || ! -f "${repo_root}/pyproject.toml" ]]; then
    echo "REPO_ROOT is unset and SLURM_SUBMIT_DIR is not the repository root" >&2
    echo "Submit from the repository root or export REPO_ROOT explicitly" >&2
    exit 2
fi
venv="${VENV:-${repo_root}/.venv}"
if [[ ! -f "${venv}/bin/activate" ]]; then
    echo "Python environment not found at ${venv}" >&2
    echo "Export VENV or run 'make setup' before submitting" >&2
    exit 2
fi
output="${OUTPUT:-${repo_root}/results/esmc6b-smoke-${SLURM_JOB_ID:-local}}"

cd "${repo_root}"
source "${venv}/bin/activate"

export HF_HOME="${HF_HOME:-${SCRATCH:-${HOME}/.cache}/huggingface}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}" "${output}"

echo "job_id=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "repo_root=${repo_root}"
echo "venv=${venv}"
echo "output=${output}"
nvidia-smi

gpu_log="${output}/gpu_memory.csv"
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

time_file="${output}/process_resources.txt"
time_command="$(type -P time)"
if [[ -z "${time_command}" ]]; then
    echo "GNU time is required to measure peak host memory" >&2
    exit 2
fi
"${time_command}" -v -o "${time_file}" \
    python cluster/slurm/smoke_esmc_brca1.py \
    --model "${MODEL_ID:-biohub/ESMC-6B}" \
    --input "${BRCA1_INPUT:-imported_data/MV_BRCA1_Findlay_2018.csv}" \
    --output "${output}" \
    --num-sequences "${NUM_SEQUENCES:-100}" \
    --dtype "${DTYPE:-bf16}" \
    --min-gpu-memory-gib "${MIN_GPU_MEMORY_GIB:-24}"

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
echo "Smoke test succeeded; results are in ${output}"
