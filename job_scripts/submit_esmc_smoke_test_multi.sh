#!/bin/bash
#SBATCH --job-name=esmc_smoke_multi
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu
#SBATCH --constraint=L40|A100
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=job_scripts/logs/esmc_smoke_multi-%j.out
#SBATCH --error=job_scripts/logs/esmc_smoke_multi-%j.err

# Node-parallel ESMC smoke test.
#
# Holds one whole L40 node (--exclusive --gres=gpu:8) and fires 8
# independent Python workers, one per GPU. Each worker loads ESMC and
# embeds GFP on its own card; per-worker stdout/stderr go to separate
# log files in job_scripts/logs/.
#
# This validates the same pattern create_embedding_batch_job would use
# in node-parallel mode: each worker masks all GPUs except its own via
# CUDA_VISIBLE_DEVICES, so torch.cuda.device_count() == 1 inside the
# worker and our single-GPU .to("cuda") path runs (no device_map=auto,
# which trips on ESMC's non-persistent rotary buffer).
#
# Smaller / faster smoke variant:
#   ESMC_MODEL=biohub/ESMC-300M sbatch --constraint=C8 --mem=64G \
#       --time=00:30:00 job_scripts/submit_esmc_smoke_test_multi.sh

set -euo pipefail
mkdir -p job_scripts/logs

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export ESMDMS_TORCH_DTYPE="${ESMDMS_TORCH_DTYPE:-bfloat16}"
export ESMC_MODEL="${ESMC_MODEL:-biohub/ESMC-6B}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "warning: HF_TOKEN is unset; gated checkpoints will fail to download." >&2
fi

job_id="${SLURM_JOB_ID:-local}"
echo "host=$(hostname) job=${job_id} model=${ESMC_MODEL}"
nvidia-smi || true

# Number of workers = number of allocated GPUs. Slurm sets
# SLURM_GPUS_ON_NODE (19.05+); fall back to 8 for interactive runs.
N_GPUS="${SLURM_GPUS_ON_NODE:-8}"
echo "launching ${N_GPUS} parallel ESMC workers (one per GPU)"

pids=()
for i in $(seq 0 $((N_GPUS - 1))); do
    log_out="job_scripts/logs/esmc_smoke_multi-${job_id}-w${i}.out"
    log_err="job_scripts/logs/esmc_smoke_multi-${job_id}-w${i}.err"
    (
        export CUDA_VISIBLE_DEVICES="${i}"
        # Inside the subshell torch sees one device renumbered to cuda:0.
        python3 job_scripts/esmc_smoke_test.py
    ) >"${log_out}" 2>"${log_err}" &
    pids+=($!)
    echo "  worker ${i} pid=${pids[$i]} log=${log_out}"
done

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "worker ${i} OK"
    else
        rc=$?
        echo "worker ${i} FAILED (exit ${rc})" >&2
        status=1
    fi
done

echo "aggregate status=${status}"
exit "${status}"
