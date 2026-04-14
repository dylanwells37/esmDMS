#!/bin/bash
#SBATCH --job-name=esm_embed
#SBATCH -p dept_cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=job_outs/slurm-%j.out
#SBATCH --error=job_outs/slurm-%j.err
#SBATCH --mem=16G

# Usage:
#   sbatch submit_embed.sh <config.json> <output_file.pkl>
#
# Example:
#   sbatch embedding_scripts/submit_embed.sh embedding_scripts/embed_config_BF520.json BF520_embeddings.pkl
#
# Optional flags passed through to embed_sequences.py:
#   --embed_zeroes      embed sequences with zero pre-selection counts
#   --esm_model <name>  override the ESM model (e.g. facebook/esm2_t6_8M_UR50D)

CONFIG=${1:-"embedding_scripts/embed_config_BF520.json"}
OUTPUT_FILE=${2:-"embeddings.pkl"}
EXTRA_ARGS="${@:3}"   # any additional flags forwarded to the Python script

source ~/popDMS/esmDMS/.venv/bin/activate
cd ~/popDMS/esmDMS

SCRDIR=/scr/${SLURM_JOB_ID}
mkdir -p $SCRDIR

python embedding_scripts/embed_sequences.py \
    "$CONFIG" \
    "$OUTPUT_FILE" \
    "$SCRDIR" \
    $EXTRA_ARGS
