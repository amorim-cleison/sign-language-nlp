DASK_PORT=5696

echo "Starting scheduler..."

poetry run dask-scheduler --port ${DASK_PORT} &> ~/repos/sign-language-nlp/cluster/out/az-scheduler.out &