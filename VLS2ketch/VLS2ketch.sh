#!/bin/bash

DATASET="ogbn-product"
START_YEAR=0
TIMESTEP=10
K_VALUE=200

for R_VALUE in {1..2}; do
    echo "Running Offline_test with R=$R_VALUE and K=$K_VALUE"
    ./Initialization --dataset "$DATASET" --R "$R_VALUE" --start_year "$START_YEAR" --K "$K_VALUE"

    for (( i = START_YEAR + 1; i < START_YEAR + TIMESTEP; i++ )); do
        echo "Running processX.py with R=$R_VALUE, K=$K_VALUE and cur_year=$i"
        python processX.py --dataset "$DATASET" --R "$R_VALUE" --K "$K_VALUE" --cur_year "$i"

        echo "Running Online_test with R=$R_VALUE, K=$K_VALUE and cur_year=$i"
        ./Update --dataset "$DATASET" --R "$R_VALUE" --cur_year "$i" --K "$K_VALUE"
    done
done
