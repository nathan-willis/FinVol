#!/bin/bash

gcc -O3 -march=native -o IC_Sedimentation2026Feb11.out TwoCurrSWEsolver.c # Compile main

# Get batch index from first argument and set up batch_file
batch_index=$1
batch_file="batch_${batch_index}.txt"

# Find all PIDS to kill 
#   ps aux | grep IC_Sedim | grep -v grep

# look for number of processes the computer can run
#   sysctl -n hw.ncpu

# do
# the ampersand is the magic that runs it in parallel
# inputs go in this order N Reynolds CFL h_min sharp U_s c2init h2init

max_jobs=256 # max parallel jobs allowed
while read param1 param2; do
    ./IC_Sedimentation2026Feb11.out 28000 1000 0.1 0.0001 50. 0.01 $param1 $param2 &
    echo "$param1 $param2" >> Feb11_2026_SedimentationInitialConditionTest/progress.log
    while (( $(jobs -r | wc -l) >= max_jobs )); do
      sleep 5
    done
done < "$batch_file"

wait
echo "Batch $batch_index completed."
