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
NODE=${SLURMD_NODENAME}
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

echo "Starting command..."
cd ../../

echo "Loading singularity modules..."
module load cuda-10.2-gcc-8.3.0-nxzzh52 &&
module load singularity-3.6.2-gcc-8.3.0-quskioo &&

echo "Executing task..."
echo " > GPUs available: ${CUDA_VISIBLE_DEVICES}"
singularity exec --nv ~/containers/openpose.sif poetry run python main.py --config ${CONFIG_FILE} --n_jobs=-1 --dask "{ 'node': '${NODE}', 'cpus_per_task': '${CPUS_PER_TASK}' }" --debug True --grid_args "{ 'lr': [0.1], 'model_args': { 'embedding_size': [128], 'hidden_size': [256], 'num_layers': [2], 'dropout': [0.1], 'num_heads': [4] }}" &&

echo "Command finished."
