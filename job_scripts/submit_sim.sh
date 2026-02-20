#!/bin/bash
#SBATCH --job-name=esm_sim
#SBATCH -p dept_cpu    
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1    
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

source ~/popDMS/esmDMS/.venv/bin/activate
cd ~/popDMS/esmDMS
python run_sim.py