#!/bin/bash

if [ ! -d "gemm/separatedConfigs" ]; then
    echo "Error: run separateConfigs.py first"
    exit 1
fi

if [ ! -d "gemm/tunedData" ]; then
    mkdir -p "gemm/tunedData"
fi

for i in {1..307}
do
    python3 ../performance/tuningRunner.py --operation gemm --configs_file=gemm/separatedConfigs/one_config_$i -d --output=gemm/tunedData/tuned_gemm_$i
done
