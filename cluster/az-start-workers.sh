SCHEDULER="20.165.54.123:5696"
DASK_SOURCE="dist/source.zip"
THREADS=2
NAME=$(hostname)
export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False


cd ~/repos/sign-language-nlp/

echo "Initializing workers..."
IFS=',' read -ra GPU <<< "${CUDA_VISIBLE_DEVICES}"
for i in "${GPU[@]}"; do
  echo " - Worker for GPU ${i}... (in background)"
  CUDA_VISIBLE_DEVICES=$i poetry run dask-worker "${SCHEDULER}" --name "worker-${NAME}-gpu${i}" --nworkers 1 --nthreads ${THREADS} &> cluster/out/az-worker-gpu${i}.out &
  # CUDA_VISIBLE_DEVICES=$i DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False singularity exec --nv ~/containers/openpose.sif poetry run dask-cuda-worker "${DASK_HOST}:${DASK_PORT}" &
done
