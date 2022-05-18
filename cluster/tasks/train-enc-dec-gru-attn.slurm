#!/bin/bash
#SBATCH --job-name=TR-EDGRUA
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=train-enc-dec-gru-attn.out

# Aqui o comando de execução de seu programa.
module load cuda-10.2-gcc-8.3.0-nxzzh52
module load singularity-3.6.2-gcc-8.3.0-quskioo

echo "Starting command..."

cd ../../
singularity exec --nv ~/containers/openpose.sif poetry run python main.py --config config/config-train-enc-dec-gru-attn.yaml

echo "Command finished."