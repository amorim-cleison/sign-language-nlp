# ---------- PARAMETERS -----------
while getopts c: flag
do
  case $flag in
    c) CONFIG_FILE=${OPTARG};;
  esac
done

validate_param () {
  if [ -z "$2" ]; then
    echo "Parameter '$1' is required"
    exit 1
  fi
}

validate_param "c" ${CONFIG_FILE}


# ---------- CODE -----------------
SLEEP_TIME=25s
NODE=${SLURMD_NODENAME}
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
DASK_SOURCE="dist/source.zip"
DASK_HOST="localhost"
DASK_PORT=$(((RANDOM % 500000) + 50000))
export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

echo "Starting command..."
cd ../../

echo "Loading singularity modules..."
module load cuda-11.0-gcc-8.3.0-fzbvcxy &&
module load singularity-3.6.2-gcc-8.3.0-quskioo &&

echo "Creating '${DASK_SOURCE}'..."
singularity exec --nv ~/containers/openpose.sif poetry build -f sdist &&
cp dist/*.whl ${DASK_SOURCE} &&

echo "Initializing scheduler..."
singularity exec --nv ~/containers/openpose.sif poetry run dask-scheduler --host ${DASK_HOST} --port ${DASK_PORT}  &

echo "Initializing workers..."
IFS=',' read -ra GPU <<< "${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}"
for i in "${GPU[@]}"; do
  echo " - Worker for GPU ${i}..."
  CUDA_VISIBLE_DEVICES=$i singularity exec --nv ~/containers/openpose.sif poetry run dask-worker "${DASK_HOST}:${DASK_PORT}" --name "Worker-GPU${i}" --nworkers 1 &
  # CUDA_VISIBLE_DEVICES=$i DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False singularity exec --nv ~/containers/openpose.sif poetry run dask-cuda-worker "${DASK_HOST}:${DASK_PORT}" &
done

echo "Waiting ${SLEEP_TIME} for workers to initialize..."
sleep ${SLEEP_TIME} &&

# echo "Executing task..."
# echo " > GPUs available: ${CUDA_VISIBLE_DEVICES}"
# singularity exec --nv ~/containers/openpose.sif poetry run python main.py --config ${CONFIG_FILE} --n_jobs=-1 --dask "{ 'node': '${NODE}', 'cpus_per_task': '${CPUS_PER_TASK}' }" &&

echo "Executing task..."
echo " > GPUs available: ${CUDA_VISIBLE_DEVICES}"
singularity exec --nv ~/containers/openpose.sif poetry run python main.py --config ${CONFIG_FILE} --n_jobs=-1 --dask "{ 'node': '${NODE}', 'cpus_per_task': '${CPUS_PER_TASK}', 'scheduler': ${DASK_HOST}:${DASK_PORT}, 'source': '${DASK_SOURCE}' }" &&

echo "Command finished."
