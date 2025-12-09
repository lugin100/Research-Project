#!/bin/bash

#SBATCH -J Inference              # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani   # Which partition will run your job
#SBATCH --time=2-0:00             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:8               # (optional) Requesting type and number of GPUs
#SBATCH --mem=30G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=slurm-output/%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=slurm-output/%j.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=luis.gindorf@student.uni-tuebingen.de   # Email to which notifications will be sent

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

# Setup Phase
# add possibly other setup code here, e.g.
# - copy singularity images or datasets to local on-compute-node storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environment variables
# - determine commandline arguments for `srun` calls

eval $(ssh-agent -s)
ssh-add ~/.ssh/github
cd $WORK/Research-Project
git fetch && git pull
source .venv/bin/activate

# Compute Phase
srun python3 Code/Inference.py  # srun will automatically pickup the configuration defined via `#SBATCH` and `sbatch` command line arguments  
