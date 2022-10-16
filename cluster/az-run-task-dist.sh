
# ---------- CODE -----------------
SCHEDULER=20.165.54.123:5696

echo "Starting command..."

cd ~/repos/sign-language-nlp/

poetry run python main.py --config config/config-transformer.yaml --n_jobs=-1 --dask "{ 'scheduler': '${SCHEDULER}' }" &> cluster/out/az-transformer-dist.out &
