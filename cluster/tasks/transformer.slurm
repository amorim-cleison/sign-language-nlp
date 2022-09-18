#!/bin/bash
#SBATCH --job-name=TRANSFOR
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=20G
#SBATCH --output=transformer.out

./../run-task.sh -c config/config-transformer.yaml
