printf 'Showing cluster status...\n'

printf '\n'
squeue

printf '\n'
sinfo -N -o "%25N %9R %14C"

printf '\n'
sinfo
