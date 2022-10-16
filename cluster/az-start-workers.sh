DASK_HOST="20.165.54.123"
DASK_PORT=5696
DASK_SOURCE="dist/source.zip"
THREADS=3
IP=$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)
export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False


cd ~/repos/sign-language-nlp/

echo "Creating '${DASK_SOURCE}'..."
poetry build -f wheel &&
cp dist/*.whl ${DASK_SOURCE} &&

echo "Initializing workers..."
IFS=',' read -ra GPU <<< "${CUDA_VISIBLE_DEVICES}"
for i in "${GPU[@]}"; do
  echo " - Worker for GPU ${i}... (in background)"
  CUDA_VISIBLE_DEVICES=$i poetry run dask-worker "${DASK_HOST}:${DASK_PORT}" --name "worker-${IP}-gpu${i}" --nworkers 1 --nthreads ${THREADS} &> ~/repos/sign-language-nlp/cluster/out/az-worker-gpu${i}.out &
  # CUDA_VISIBLE_DEVICES=$i DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False singularity exec --nv ~/containers/openpose.sif poetry run dask-cuda-worker "${DASK_HOST}:${DASK_PORT}" &
done
