#!/bin/bash

data_files=("data00001.data" "data00002.data" "data00003.data")

for data in "${data_files[@]}"; do
    base=$(basename "$data" .data)
    input_file="in_${base}.in"

    # Replace placeholder
    sed "s|__DATAFILE__|$data|" in.in > $input_file

    echo "Running $data"
    mpirun -np 50 ./lmp_mpi -in $input_file
done           
