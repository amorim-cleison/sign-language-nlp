#!/bin/bash
#SBATCH --job-name=ED-GRU
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=enc-dec-gru-attn.out

./../run-task.sh -c config/config-enc-dec-gru-attn.yaml

