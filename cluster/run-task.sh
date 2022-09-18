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
SOURCE_FILE="dist/source.zip"
DASK_HOST="localhost"
DASK_PORT=8786
SLEEP_TIME=30s

echo "Starting command..."
cd ../../

echo "Loading singularity modules..."
module load cuda-10.2-gcc-8.3.0-nxzzh52 &&
module load singularity-3.6.2-gcc-8.3.0-quskioo &&

echo "Creating '${SOURCE_FILE}'..."
singularity exec --nv ~/containers/openpose.sif poetry build -f sdist &&
cp dist/*.whl ${SOURCE_FILE} &&


echo "Initializing scheduler..."
singularity exec --nv ~/containers/openpose.sif poetry run dask-scheduler --host ${DASK_HOST} --port ${DASK_PORT}  &

echo "Initializing workers..."
# CUDA_VISIBLE_DEVICES=0 singularity exec --nv ~/containers/openpose.sif poetry run dask-worker localhost:8786 --nworkers 'auto' --name 'worker-1' &
# CUDA_VISIBLE_DEVICES=1 singularity exec --nv ~/containers/openpose.sif poetry run dask-worker localhost:8786 --nworkers 'auto' --name 'worker-2' & 
# CUDA_VISIBLE_DEVICES=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS} singularity exec --nv ~/containers/openpose.sif poetry run dask-worker localhost:8786 --nworkers 1 --name 'worker-2' &

IFS=',' read -ra GPU <<< "${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}"
for i in "${GPU[@]}"; do
  echo " - Worker for GPU ${i}..."
  CUDA_VISIBLE_DEVICES=$i singularity exec --nv ~/containers/openpose.sif poetry run dask-worker "${DASK_HOST}:${DASK_PORT}" --name "Worker-GPU${i}" --nworkers 'auto' &
done

echo "Waiting ${SLEEP_TIME} for workers to initialize..."
sleep ${SLEEP_TIME} &&

echo "Running project..."
# singularity exec --nv ~/containers/openpose.sif poetry run python main.py --config ${CONFIG_FILE} --n_jobs=-1 --source_file=${SOURCE_FILE} &&
singularity exec --nv ~/containers/openpose.sif poetry run python main.py --config ${CONFIG_FILE} --n_jobs=-1 &&

echo "Command finished."
