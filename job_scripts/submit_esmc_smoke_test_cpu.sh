#!/bin/bash
#SBATCH --job-name=esmc_smoke_cpu
#SBATCH --partition=big_memory
#SBATCH --cpus-per-task=4
#SBATCH --mem=56G
#SBATCH --time=01:00:00
#SBATCH --output=job_scripts/logs/esmc_smoke_cpu-%j.out
#SBATCH --error=job_scripts/logs/esmc_smoke_cpu-%j.err

# CPU-only smoke test for ESMC on the big_memory partition.
#   Uses float32 on CPU (bfloat16 is a no-op on most CPUs).
#   Memory: 56 G allocated (6B model weights ~48 GB in fp32).
#   For the smaller 300M model you can drop --mem to 16G and --time to 00:20:00.

set -euo pipefail
mkdir -p job_scripts/logs

source /opt/mamba/etc/profile.d/conda.sh && conda activate py312

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Force CPU; no bfloat16 needed (float32 is the safe default on CPU).
export ESMDMS_TORCH_DTYPE="${ESMDMS_TORCH_DTYPE:-float32}"

export ESMC_MODEL="${ESMC_MODEL:-biohub/ESMC-6B}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "warning: HF_TOKEN is unset; gated checkpoints will fail to download." >&2
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"

python3 job_scripts/esmc_smoke_test_cpu.py
