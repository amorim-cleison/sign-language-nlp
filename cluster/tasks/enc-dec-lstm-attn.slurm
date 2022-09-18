#!/bin/bash
#SBATCH --job-name=ED-LSTM
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=enc-dec-lstm-attn.out

./../run-task.sh -c config/config-enc-dec-lstm-attn.yaml.yaml
