#!/bin/bash

if [ ! -d "conv/separatedConfigs" ]; then
    echo "Error: run separateConfigs.py first"
    exit 1
fi

if [ ! -d "conv/tunedData" ]; then
    mkdir -p "conv/tunedData"
fi

for i in {1..64}
do
    python3 ../performance/tuningRunner.py --operation conv --configs_file= --configs_file=conv/separatedConfigs/one_config_$i -d --output=conv/tunedData/tuned_conv_$i
done
