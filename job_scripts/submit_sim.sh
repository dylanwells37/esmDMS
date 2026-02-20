#!/bin/bash
#SBATCH --job-name=esm_sim
#SBATCH -p dept_cpu    
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=job_outs/slurm-%j.out

###SBATCH --mem=2G


source ~/popDMS/esmDMS/.venv/bin/activate
cd ~/popDMS/esmDMS
python run_sim.py job_settings/config_plus1.json
