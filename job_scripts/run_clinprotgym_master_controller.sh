#!/bin/bash
#SBATCH --job-name=clinpgym_master
#SBATCH -p any_cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --output=job_outs/clinpgym-master-%j.out
#SBATCH --error=job_outs/clinpgym-master-%j.err

set -euo pipefail

# One-shot ClinProtGym ESM-C SAE controller.
#
# Submit from the repo root:
#   sbatch job_scripts/run_clinprotgym_master_controller.sh
#
# The Python controller submits the next check with --begin=now+Nminutes
# whenever it is waiting for array jobs to finish.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p job_outs

export PYTHONUNBUFFERED=1

python3 scripts/clinprotgym_master_controller.py \
  --controller-script "$REPO_ROOT/job_scripts/run_clinprotgym_master_controller.sh" \
  --resubmit-self \
  "$@"
