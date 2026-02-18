#!/bin/bash 
 
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name=protein_embeddings

# partition (queue) declaration
# Department choices: dept_cpu, dept_gpu, any_cpu, any_gpu, big_memory
# Group choices: bahar_gpu, benos, benos_gpu, camacho_gpu, chakra_24, chakra_gpu
#SBATCH --partition=any_cpu

# number of requested nodes
#SBATCH --nodes=1

# number of tasks
#SBATCH --ntasks=1

# number of requested cores
#SBATCH --ntasks-per-node=1


# call a Slurm Feature
### For multiple selections, seperate using the pipe "|"

# requested runtime
# #SBATCH --time=05:00:00 

# standard output & error
# #SBATCH --error=std.err
# #SBATCH --output=std.out

# send email about job start and end
# #SBATCH --mail-user=dhw28@pitt.edu
# #SBATCH --mail-type=ALL

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     User Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

### Create a temp scratch directory on local node hardrive
# current (working) direcotry
work_dir=${SLURM_SUBMIT_DIR}

# username
user=$(whoami)

# directory name where job will be run (on compute node)
job_dir="${user}_${SLURM_JOB_ID}.dcb.private.net"

# creating directory on /scr folder of compute node
mkdir /scr/$job_dir

# change to the newly created directory
cd /scr/$job_dir

# copy the submit file (and all other related files/directories)
rsync -a ${SLURM_SUBMIT_DIR}/* .

### Some documentation (date and node)
# put date and time of starting job in a file
date > date.txt

# put hostname of compute node in a file
hostname > hostname.txt

# copy files on exit or interrupt
# make sure this is before your main program for it to always run on exit
trap "echo 'copying files'; rsync -avz * ${SLURM_SUBMIT_DIR}" EXIT

# run your program here
# the below example runs a cpu stress test
stress-ng --cpu $SLURM_TASKS_PER_NODE --timeout 120s --metrics-brief > stress-ng.log

# append date and time of finished job in a file
date >> date.txt

# Leave this line to tell slurm that the script finished correctly
exit 0
