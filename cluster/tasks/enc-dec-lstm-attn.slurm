#!/bin/bash
#SBATCH --job-name=ED-LSTM
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=enc-dec-lstm-attn.out
#SBATCH --nodelist=node004
#SBATCH --partition=long_gpu

./../run-task.sh -c config/config-enc-dec-lstm-attn.yaml
