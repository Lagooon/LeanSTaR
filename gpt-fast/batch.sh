#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=result.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=512G
#SBATCH -c 192
#SBATCH --gres=gpu:8
#SBATCH -p sched_oliva
#SBATCH --qos=sched_mit_newuser
srun bash scripts_intern/finetune_7b_intern.sh
