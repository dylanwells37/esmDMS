#!/bin/bash
#SBATCH --job-name=esm_sim
#SBATCH -p dept_cpu    
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=job_outs/slurm-%j.out
#SBATCH --error=job_outs/slurm-%j.err
#SBATCH --mem=1G


source ~/popDMS/esmDMS/.venv/bin/activate
cd ~/popDMS/esmDMS

SCRDIR=/scr/${SLURM_JOB_ID}

# find the number of job in the batch
job_number=$(echo $SLURM_ARRAY_TASK_ID)

python run_sim.py job_settings/config_plus1.json n_rep_error_bars_layer${SLURM_ARRAY_TASK_ID}.pkl $SCRDIR --layer $SLURM_ARRAY_TASK_ID
