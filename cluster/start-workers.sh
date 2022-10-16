SCHEDULER="20.165.54.123:5696"
DASK_SOURCE="dist/source.zip"
THREADS=3
NAME=$(hostname)
export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False


cd ~/repos/sign-language-nlp/

echo "Loading singularity modules..."
module load cuda-11.0-gcc-8.3.0-fzbvcxy
module load singularity-3.6.2-gcc-8.3.0-quskioo

echo "Initializing workers..."
IFS=',' read -ra GPU <<< "${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}"
for i in "${GPU[@]}"; do
  echo " - Worker for GPU ${i}... (in background)"
  CUDA_VISIBLE_DEVICES=$i DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False singularity exec --nv ~/containers/openpose.sif poetry run dask-worker "${SCHEDULER}" --name "worker-${NAME}-gpu${i}" --nworkers 1 --nthreads "${THREADS}" &> cluster/out/worker-gpu${i}.out &
done

sleep 20s

tail -f cluster/out/*worker*.out
