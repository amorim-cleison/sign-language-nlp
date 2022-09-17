#!/bin/bash
#SBATCH --job-name=ED-GRU
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=enc-dec-gru-attn.out

# Aqui o comando de execução de seu programa.
module load cuda-10.2-gcc-8.3.0-nxzzh52
module load singularity-3.6.2-gcc-8.3.0-quskioo

echo "Starting command..."

cd ../../
singularity exec --nv ~/containers/openpose.sif poetry run python main.py --config config/config-enc-dec-gru-attn.yaml --n_jobs=2

echo "Command finished."
