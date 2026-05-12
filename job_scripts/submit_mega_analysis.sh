#!/bin/bash
#SBATCH --job-name=mega_analysis
#SBATCH -p dept_cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=job_outs/slurm-%j.out
#SBATCH --error=job_outs/slurm-%j.err
#SBATCH --mem=24G

# Usage:
#   sbatch job_scripts/submit_mega_analysis.sh <dataset> <output_dir> [--sim_config <cfg>] [--embedding_config <cfg>] [--all | --fitness | --sel_coeffs | ...]
#
#   <dataset> is the dataset name (e.g. Ube4b).  It is substituted for {dataset}
#   in all paths inside the config files, so no paths need to be edited per run.
#
# Examples:
#   sbatch job_scripts/submit_mega_analysis.sh Ube4b plots/ --sim_config configs/simulation_config.json --fitness
#   sbatch job_scripts/submit_mega_analysis.sh Ube4b plots/ --embedding_config configs/inference_config.json --cross_replicate_consistency --normalize by_layer_dim --force_recompute
#   sbatch job_scripts/submit_mega_analysis.sh Ube4b plots/ --sim_config configs/simulation_config.json --embedding_config configs/inference_config.json --all
#   sbatch job_scripts/submit_mega_analysis.sh Ube4b plots/ --embedding_config configs/inference_config.json --popDMS_comparison
#
# Analysis flags (pass one or more, or --all):
#   --fitness                      true vs inferred fitness
#   --sel_coeffs                   true vs inferred selection coefficients
#   --cross_replicate_consistency  cross-replicate consistency of inferred sel coeffs
#   --shuffled_frequencies         cross-replicate consistency with shuffled frequencies
#   --popDMS_comparison            ESM-DMS vs popDMS fitness + enrichment ratio
#   --all                          run all implemented analyses
#
# Optional flags:
#   --normalize none|by_layer|by_layer_dim   z-normalize embeddings before inference (default: none)
#   --force_recompute                        ignore cached inference_results.pkl
#   --replicates 1 2 3                       restrict inference to a subset of replicate IDs

DATASET=${1:?"Usage: sbatch submit_mega_analysis.sh <dataset> <output_dir> [extra args]"}
OUTPUT_DIR=${2:?"Usage: sbatch submit_mega_analysis.sh <dataset> <output_dir> [extra args]"}
EXTRA_ARGS="${@:3}"   # any additional flags forwarded to the Python script

source ~/popDMS/esmDMS/.venv/bin/activate
cd ~/popDMS/esmDMS

SCRDIR=/scr/${SLURM_JOB_ID}
mkdir -p $SCRDIR

export PYTHONUNBUFFERED=1

echo "Running mega_analysis with the following parameters:"
echo "DATASET:     $DATASET"
echo "OUTPUT_DIR:  $OUTPUT_DIR"
echo "EXTRA_ARGS:  $EXTRA_ARGS"

python mega_analysis.py \
    "$DATASET" \
    "$OUTPUT_DIR" \
    $EXTRA_ARGS


# echo time taken and max memory used
echo "Job completed in $(($SECONDS / 3600)) hours $((($SECONDS % 3600) / 60)) minutes and $(($SECONDS % 60)) seconds."
#echo "Max memory used: $(sacct -j ${SLURM_JOB_ID} --format=MaxRSS --noheader | awk '{print $1}')"