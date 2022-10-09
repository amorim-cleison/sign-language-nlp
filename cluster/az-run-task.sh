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
NODE=localhost
CPUS_PER_TASK=6
# export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

echo "Starting command..."
cd ../../

echo "Executing task..."
echo " > GPUs available: ${CUDA_VISIBLE_DEVICES}"
poetry run python main.py --config ${CONFIG_FILE} --n_jobs=-1 --dask "{ 'node': '${NODE}', 'cpus_per_task': '${CPUS_PER_TASK}' }" &&

echo "Command finished."
