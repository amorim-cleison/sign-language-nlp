#!/bin/bash
#SBATCH --job-name=TRANSFOR
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=transformer.out

# Aqui o comando de execução de seu programa.
module load cuda-10.2-gcc-8.3.0-nxzzh52
module load singularity-3.6.2-gcc-8.3.0-quskioo

echo "Starting command..."

cd ../../
singularity exec --nv ~/containers/openpose.sif poetry run python main.py --config config/config-transformer.yaml --n_jobs=2

echo "Command finished."
