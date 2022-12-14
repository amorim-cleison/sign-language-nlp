#!/bin/bash
#SBATCH --job-name=TRANSFOR
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --output=transformer.out
##SBATCH --nodelist=node005
#SBATCH --partition=long_gpu

./../run-task-old.sh -c config/config-transformer.yaml
