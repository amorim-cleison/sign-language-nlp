mkdir -p out/

echo 'Submitting all tasks...'
(cd out/ && for FILE in ../tasks/*.slurm; do sbatch -p long_gpu $FILE; done)

sleep 2s
squeue

sleep 5s
./tail-all.sh
