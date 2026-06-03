#!/bin/bash
#SBATCH --job-name=esmc_smoke
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=C8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=job_scripts/logs/esmc_smoke-%j.out
#SBATCH --error=job_scripts/logs/esmc_smoke-%j.err

# Quick GPU + ESMC smoke test.
#   This cluster uses Slurm features (not typed GRES) for GPU model:
#     --constraint=C8           any Ampere+ (sm_80+, bf16 capable)
#     --constraint=L40          48 GB L40 (32 cards, g020-g023)
#     --constraint=A100         40 GB A100 (4 dept_gpu cards on g019)
#     --constraint='L40|A100'   either 40+ GB Ampere+
#   Defaults to biohub/ESMC-300M + --constraint=C8 (smallest viable Ampere+).
#   For the 6B model:
#     sbatch --constraint='L40|A100' --mem=64G --time=04:00:00 \
#         --export=ALL,ESMC_MODEL=biohub/ESMC-6B job_scripts/submit_esmc_smoke_test.sh

set -euo pipefail
mkdir -p job_scripts/logs

# Pin HF cache to a shared location so we don't redownload per job. Edit
# this path to point at your own scratch/home share.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Use bfloat16 by default on CUDA (required for 6B to fit; harmless for 300M).
export ESMDMS_TORCH_DTYPE="${ESMDMS_TORCH_DTYPE:-bfloat16}"

# Set ESMC_MODEL=biohub/ESMC-6B to actually exercise the 6B path.
export ESMC_MODEL="${ESMC_MODEL:-biohub/ESMC-300M}"

# HF_TOKEN must be exported in your shell environment for gated models.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "warning: HF_TOKEN is unset; gated checkpoints will fail to download." >&2
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi || true

python3 job_scripts/esmc_smoke_test.py
