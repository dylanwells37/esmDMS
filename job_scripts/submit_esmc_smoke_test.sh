#!/bin/bash
#SBATCH --job-name=esmc_smoke
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=job_scripts/logs/esmc_smoke-%j.out
#SBATCH --error=job_scripts/logs/esmc_smoke-%j.err

# Quick GPU + ESMC smoke test.
#   Defaults to biohub/ESMC-300M (fits on any 12 GB+ card, ~30s warm).
#   For the 6B model on this cluster, request a 48 GB L40 or 40 GB A100:
#     #SBATCH --gres=gpu:l40:1     (32 cards available)
#     #SBATCH --gres=gpu:a100:1    (4 cards available)
#   and set ESMC_MODEL=biohub/ESMC-6B + ESMDMS_TORCH_DTYPE=bfloat16.

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
