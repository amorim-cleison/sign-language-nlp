mkdir -p out/

printf 'Submitting all tasks...\n'
(cd out/ && for FILE in ../tasks/*.slurm; do printf ' Submitting %s... ' "$FILE"; sbatch -p long_gpu $FILE; sleep 15s; done)

printf '\n'
squeue

printf '\n'
./tail-all.sh
