#!/bin/bash
#SBATCH --job-name=TRNF-WRKS
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --output=out/launch-workers.out
#SBATCH --nodelist=node004
#SBATCH --partition=long_gpu
 
cd ~/repos/sign-language-nlp/

./cluster/start-workers.sh
