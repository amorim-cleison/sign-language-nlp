#!/bin/bash
#SBATCH --job-name=DIST
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --mem=20G
#SBATCH --output=distributed.out

# Aqui o comando de execução de seu programa.
module load cuda-10.2-gcc-8.3.0-nxzzh52
module load singularity-3.6.2-gcc-8.3.0-quskioo

echo "Starting command..."

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=12355

cd ../../
singularity exec --nv  ~/containers/openpose.sif poetry run python main.py --config config/config-transformer.yaml --n_jobs=2

echo "Command finished."
