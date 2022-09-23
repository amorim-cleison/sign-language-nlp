mkdir -p out/

printf 'Deleting previous outputs...\n'
rm -f /out/*.out

printf 'Deleting cached datasets...\n'
rm -f /tmp/*.dataset.tmp

printf 'Submitting tasks (with interval)...\n'
(cd out/ && for FILE in ../tasks/*.slurm; do printf ' %s -> ' "$FILE"; sbatch $FILE; sleep 5s; done;)

printf '\n'
./status-cluster.sh

printf '\n'
./tail-all.sh
