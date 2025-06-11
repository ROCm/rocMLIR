#!/bin/bash

# Arguments: $1 = WORKSPACE, $2 = MITuna path
set -Eeuo pipefail

cd build

../mlir/utils/tuna/tuna-script.sh -o gemm        \
    -c ../mlir/utils/jenkins/ci-configs/selected-gemm-configs \
    -t "$2" -f tuning_gemm.tsv
test -f tuning_gemm.tsv

../mlir/utils/tuna/tuna-script.sh -o convolution \
    -c ../mlir/utils/jenkins/ci-configs/selected-conv-configs \
    -t "$2" -f tuning_conv.tsv
test -f tuning_conv.tsv

../mlir/utils/tuna/tuna-script.sh -o attention   \
    -c ../mlir/utils/jenkins/ci-configs/selected-attention-configs \
    -t "$2" -f tuning_attention.tsv
test -f tuning_attention.tsv

# quick sweeps
../mlir/utils/tuna/tuna-script.sh -o gemm        \
    -c ../mlir/utils/jenkins/ci-configs/selected-gemm-configs \
    -t "$2" -f quick_tuning_gemm.tsv -s quick
test -f quick_tuning_gemm.tsv

../mlir/utils/tuna/tuna-script.sh -o convolution \
    -c ../mlir/utils/jenkins/ci-configs/selected-conv-configs \
    -t "$2" -f quick_tuning_conv.tsv -s quick
test -f quick_tuning_conv.tsv
