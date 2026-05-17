#!/bin/bash
#SBATCH --job-name=mega_analysis
#SBATCH -p any_cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=job_outs/slurm-%j.out
#SBATCH --error=job_outs/slurm-%j.err
#SBATCH --mem=24G

# Usage:
#   sbatch job_scripts/submit_mega_analysis.sh <dataset> <output_dir> [--sim_config <cfg>] [--embedding_config <cfg>] [--all | --fitness | ...]
#
#   <dataset> is the primary dataset name (e.g. Ube4b).  It is substituted for
#   {dataset} in all paths inside the config files.
#
#   For analyses that require multiple datasets (--shuffled_consistency), set the
#   DATASETS environment variable to a space-separated list of dataset names.
#   The first name should match <dataset> (used for output path naming).
#
# Single-dataset examples:
#   sbatch job_scripts/submit_mega_analysis.sh Ube4b plots/ --sim_config configs/simulation_config.json --fitness
#   sbatch job_scripts/submit_mega_analysis.sh Ube4b plots/ --embedding_config configs/inference_config.json --cross_replicate_consistency --every_n 5
#   sbatch job_scripts/submit_mega_analysis.sh Ube4b plots/ --embedding_config configs/inference_config.json --shuffled_frequencies --normalize by_layer_dim
#   sbatch job_scripts/submit_mega_analysis.sh Ube4b plots/ --sim_config configs/simulation_config.json --embedding_config configs/inference_config.json --all
#   sbatch job_scripts/submit_mega_analysis.sh Ube4b plots/ --embedding_config configs/inference_config.json --popDMS_comparison
#
# Multi-dataset example (shuffled_consistency):
#   DATASETS="BRCA1 Ube4b TpoR BF520 BG505" sbatch job_scripts/submit_mega_analysis.sh BRCA1 plots/ --embedding_config configs/inference_config.json --shuffled_consistency
#   # Or with SBATCH env passthrough:
#   sbatch --export=ALL,DATASETS="BRCA1 Ube4b TpoR BF520 BG505" job_scripts/submit_mega_analysis.sh BRCA1 plots/ --embedding_config configs/inference_config.json --shuffled_consistency
#
# Analysis flags (pass one or more, or --all):
#   --fitness                      true vs inferred fitness
#   --sel_coeffs                   true vs inferred selection coefficients
#   --cross_replicate_consistency  cross-replicate consistency of inferred sel coeffs across layers
#   --shuffled_frequencies         cross-replicate consistency with shuffled frequencies (per dataset)
#   --shuffled_consistency         compare shuffled correlations across multiple datasets
#   --popDMS_comparison            ESM-DMS vs popDMS fitness + enrichment ratio
#   --all                          run all implemented analyses
#
# Optional flags:
#   --normalize none|by_layer|by_layer_dim   z-normalize embeddings before inference (default: none)
#   --every_n <N>                            only plot per-layer plots every N layers (default: 1)
#   --force_recompute                        ignore cached inference_results.pkl
#   --replicates 1 2 3                       restrict inference to a subset of replicate IDs

DATASET=${1:?"Usage: sbatch submit_mega_analysis.sh <dataset> <output_dir> [extra args]"}
OUTPUT_DIR=${2:?"Usage: sbatch submit_mega_analysis.sh <dataset> <output_dir> [extra args]"}
EXTRA_ARGS="${@:3}"   # any additional flags forwarded to the Python script

# DATASETS defaults to the single primary dataset. Override via env var for
# multi-dataset analyses (e.g. DATASETS="Ube4b Ube4b_v2" sbatch ...).
DATASETS="${DATASETS:-$DATASET}"

source ~/popDMS/esmDMS/.venv/bin/activate
cd ~/popDMS/esmDMS

SCRDIR=/scr/${SLURM_JOB_ID}
mkdir -p $SCRDIR

export PYTHONUNBUFFERED=1

echo "Running mega_analysis with the following parameters:"
echo "DATASETS:    $DATASETS"
echo "OUTPUT_DIR:  $OUTPUT_DIR"
echo "EXTRA_ARGS:  $EXTRA_ARGS"

# NOTE: $DATASETS is intentionally unquoted so multiple names are passed as
# separate positional arguments to the Python CLI.
python mega_analysis.py \
    $DATASETS \
    "$OUTPUT_DIR" \
    $EXTRA_ARGS

echo "Job completed in $(($SECONDS / 3600)) hours $((($SECONDS % 3600) / 60)) minutes and $(($SECONDS % 60)) seconds."
