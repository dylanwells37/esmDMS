#!/bin/bash
#SBATCH --job-name=mega_analysis
#SBATCH -p dept_cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=job_outs/slurm-%j.out
#SBATCH --error=job_outs/slurm-%j.err
#SBATCH --mem=24G

# Usage:
#   sbatch job_scripts/submit_mega_analysis.sh <embedding_path> <output_dir> [--sim_config <cfg>] [--embedding_config <cfg>] [--all | --fitness | --sel_coeffs | ...]
#
# Examples:
#   sbatch job_scripts/submit_mega_analysis.sh ~/popDMS/esmDMS/data/sequence_data/BRCA1 plots/ --sim_config configs/simulation_config.json --fitness
#   sbatch job_scripts/submit_mega_analysis.sh ~/popDMS/esmDMS/data/sequence_data/BRCA1 plots/ --embedding_config configs/inference_config.json --cross_replicate_consistency
#   sbatch job_scripts/submit_mega_analysis.sh ~/popDMS/esmDMS/data/sequence_data/BRCA1 plots/ --sim_config configs/simulation_config.json --embedding_config configs/inference_config.json --all
#
# Analysis flags (pass one or more, or --all):
#   --fitness                      true vs inferred fitness
#   --sel_coeffs                   true vs inferred selection coefficients
#   --cross_replicate_consistency  cross-replicate consistency of inferred sel coeffs
#   --shuffled_frequencies         cross-replicate consistency with shuffled frequencies
#   --all                          run all implemented analyses

EMBEDDING_PATH=${1:?"Usage: sbatch submit_mega_analysis.sh <embedding_path> <output_dir> [extra args]"}
OUTPUT_DIR=${2:?"Usage: sbatch submit_mega_analysis.sh <embedding_path> <output_dir> [extra args]"}
EXTRA_ARGS="${@:3}"   # any additional flags forwarded to the Python script

source ~/popDMS/esmDMS/.venv/bin/activate
cd ~/popDMS/esmDMS

SCRDIR=/scr/${SLURM_JOB_ID}
mkdir -p $SCRDIR

echo "Running mega_analysis with the following parameters:"
echo "EMBEDDING_PATH: $EMBEDDING_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EXTRA_ARGS: $EXTRA_ARGS"

python mega_analysis.py \
    "$EMBEDDING_PATH" \
    "$OUTPUT_DIR" \
    $EXTRA_ARGS
