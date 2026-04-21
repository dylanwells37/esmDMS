#!/bin/bash
#SBATCH --job-name=esm_embed_arr
#SBATCH -p dept_cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=job_outs/slurm-%A_%a.out
#SBATCH --error=job_outs/slurm-%A_%a.err
#SBATCH --mem=16G

# Usage:
#   sbatch --array=0-N embedding_scripts/submit_embed_array.sh <config.json> <output_file.pkl> [N+1]
#
# The third argument (n_chunks) must equal the array size (N+1 in --array=0-N).
# It is passed to the Python script so each job knows the total chunk count.
#
# Examples — embed BF520 sequences across 10 jobs:
#   sbatch --array=0-9 embedding_scripts/submit_embed_array.sh \
#       embedding_scripts/embed_config_BF520.json BF520_embeddings.pkl 10
#
# Optional flags forwarded to embed_sequences.py:
#   --embed_zeroes      embed sequences with zero pre-selection counts
#   --esm_model <name>  override the ESM model

CONFIG=${1:-"embedding_scripts/embed_config_BF520.json"}
OUTPUT_FILE=${2:-"embeddings.pkl"}
N_CHUNKS=${3:-10}
EXTRA_ARGS="${@:4}"


SCRDIR=/scr/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRDIR
export TMPDIR=$SCRDIR

source ~/popDMS/esmDMS/.venv/bin/activate
cd ~/popDMS/esmDMS

python embedding_scripts/embed_sequences.py \
    "$CONFIG" \
    "$OUTPUT_FILE" \
    "$SCRDIR" \
    --chunk_idx $SLURM_ARRAY_TASK_ID \
    --n_chunks $N_CHUNKS \
    $EXTRA_ARGS
