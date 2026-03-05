#!/bin/bash

gcc -O3 -march=native -o DepositionExamplePlots_Mar3.out TwoCurrSWEsolver.c # Compile main

# Get batch index from first argument and set up batch_file
batch_file="$1"

# Find all PIDS to kill 
#   ps aux | grep DepositionExamplePlots_Mar3| grep -v grep

# look for number of processes the computer can run
#   sysctl -n hw.ncpu

# do
# the ampersand is the magic that runs it in parallel
# inputs go in this order N Reynolds CFL h_min sharp U_s c2init h2init

# nohup ./IC_batch_runs.sh DepositExamplePlots_batch.txt &

max_jobs=12 # max parallel jobs allowed
while read param1 param2; do
    ./DepositionExamplePlots_Mar3.out 28000 1000 0.1 0.0001 200. 0.01 $param1 $param2 &
    echo "$param1 $param2" >> Mar3_DepositionExamplePlots/progress.log
    while (( $(jobs -r | wc -l) >= max_jobs )); do
      sleep 5
    done
done < "$batch_file"

wait
echo "Batch $batch_index completed."
