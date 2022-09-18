printf 'Showing cluster status...\n'

printf '\n'
sinfo -N -o "%25N %9R %14C"

printf '\n'
sinfo

printf '\n'
squeue -u cca5

printf '\n'