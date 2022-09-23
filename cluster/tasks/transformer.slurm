#!/bin/bash
#SBATCH --job-name=TRANSFOR
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=transformer.out
#SBATCH --nodelist=node004
#SBATCH --partition=long_gpu

./../run-task.sh -c config/config-transformer.yaml
