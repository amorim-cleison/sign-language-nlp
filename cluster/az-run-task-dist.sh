
# ---------- CODE -----------------
SCHEDULER=20.165.54.123:5696
DASK_SOURCE="dist/source.zip"

echo "Starting command..."

cd ~/repos/sign-language-nlp/


echo "Creating '${DASK_SOURCE}'..."
poetry build -f sdist &&
cp dist/*.whl ${DASK_SOURCE} &&


echo "Executing task..."
poetry run python main.py --config config/config-transformer.yaml --n_jobs=-1 --dask "{ 'scheduler': '${SCHEDULER}', 'source': '${DASK_SOURCE}' }" &> cluster/out/az-transformer-dist.out &
