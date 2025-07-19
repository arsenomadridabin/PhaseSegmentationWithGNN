#!/bin/bash
#SBATCH -N 1                    # number of nodes
##SBATCH -n 4			# number of MPI processes
##SBATCH -c 12			# 6 threads per MPI process
#SBATCH -t 15:00:00
#SBATCH -p gpu
#SBATCH -A hpc_bb_karki3
#SBATCH -o  gb.out
#SBATCH -e  err.out

source ~/miniconda3/etc/profile.d/conda.sh

# Activate env
conda activate dpai_env

# Training
#python -u train.py > train.log 2>&1

#Evaluate
python -u evaluate_updated.py > eval.log 2>&1
