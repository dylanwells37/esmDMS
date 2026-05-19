#!/bin/bash
#SBATCH --job-name=esm_embed
#SBATCH -p dept_cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=job_outs/slurm-%j.out
#SBATCH --error=job_outs/slurm-%j.err
#SBATCH --mem=4G

# Usage:
#   sbatch submit_embed.sh <config.json> <output_file.pkl>
#
# Example:
#   sbatch embedding_scripts/submit_embed.sh embedding_scripts/embed_config_BF520.json BF520_embeddings.pkl
#
# Optional flags passed through to embed_sequences.py:
#   --embed_zeroes      embed sequences with zero pre-selection counts
#   --esm_model <name>  override the ESM model (e.g. facebook/esm2_t6_8M_UR50D)
#   --embedding_method <mean_pool|cls|mutation_site>
#   --pool_mutations    average multiple mutation-site embeddings into one vector

CONFIG=${1:-"embedding_scripts/embed_config_BF520.json"}
OUTPUT_FILE=${2:-"embeddings.pkl"}
EXTRA_ARGS="${@:3}"   # any additional flags forwarded to the Python script





# 1. Check if /tmp is noexec
findmnt /tmp | grep noexec

# 2. Check your ulimit for virtual memory
ulimit -v

# 3. Verify TMPDIR is actually being set in the job (add to your script temporarily)
echo "TMPDIR=$TMPDIR"
echo "SCRDIR=$SCRDIR"
findmnt $SCRDIR

# 4. Check PyTorch/CUDA version mismatch
source ~/popDMS/esmDMS/.venv/bin/activate
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
nvidia-smi  # may show nothing on cpu node, that's fine

# 5. Check if the .so file is actually readable/intact
ls -lh ~/popDMS/esmDMS/.venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so




source ~/popDMS/esmDMS/.venv/bin/activate
cd ~/popDMS/esmDMS

SCRDIR=/scr/${SLURM_JOB_ID}
mkdir -p $SCRDIR



python embedding_scripts/embed_sequences.py \
    "$CONFIG" \
    "$OUTPUT_FILE" \
    "$SCRDIR" \
    $EXTRA_ARGS

