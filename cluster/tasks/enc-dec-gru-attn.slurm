#!/bin/bash
#SBATCH --job-name=ED-GRU
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=enc-dec-gru-attn.out
#SBATCH --nodelist=node005
#SBATCH --partition=long_gpu

./../run-task.sh -c config/config-enc-dec-gru-attn.yaml

