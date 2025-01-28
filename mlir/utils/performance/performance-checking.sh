#!/bin/bash

# Shell script that captures the performance difference between data types to validate expected kernel performance.
# Usage: /performance-checking --d <model> --p <model_path> [--r <number_of_iterations>]"
# Arguments:
#       --d <model>                 Used model, will be the name for the directory with kernels (default: 'resnet50-fp16').
#       --p <model_path>            Path to .onnx file (default: '/models/mlperf/resnet50_v1.onnx').
#       --r <number_of_iterations>  Number of times to run each testcase (default: 5).

MODEL_NAME="resnet50-fp16"
MODEL_PATH="/models/mlperf/resnet50_v1.onnx"
RUNS=5

while [[ $# -gt 0 ]]; do
        case "$1" in
        --d)
                MODEL_NAME="$2"
                shift 2
                ;;
        --p)
                MODEL_PATH="$2"
                shift 2
                ;;
        --r)
                RUNS="$2"
                shift 2
                ;;
        --help)
                echo "Usage: $0 --d <model> --p <model_path> [--r <number_of_iterations>]"
                exit
                ;;
        *)
                        echo "Option $1 doesn't exist"
                        exit 1
                        ;;
        esac
done


mkdir "$MODEL_NAME"

MIGRAPHX_MLIR_DUMP_TO_MXR="$MODEL_NAME" MIGRAPHX_ENABLE_NHWC=1 MIGRAPHX_ENABLE_HIPBLASLT_GEMM=1 MIGRAPHX_MLIR_USE_SPECIFIC_OPS="convolution,~fused,~dot" migraphx-driver compile "$MODEL_PATH" --fp16 --exhaustive-tune

ls "$MODEL_NAME"/*.mxr |xargs -I {} -n 1 migraphx-driver read "{}" --py -o "{}".py

sed -i -e 's/half_type/int8_type/' -e 's/convolution/quant_convolution/' "$MODEL_NAME"/*.py

echo "NEW RUN" >> "$MODEL_NAME/times"

for testcase in "$MODEL_NAME"/*.py
do
        test_name=$(basename "$testcase")
        total_time=0

        for ((i = 1; i <= RUNS; i++))
        do
                MIGRAPHX_DISABLE_PASSES=auto_contiguous migraphx-driver time $testcase --mlir > "$MODEL_NAME/results.out"
                #run_time=$(awk -F'[/ ]' -v t="$testcase" '/Total time/{k=$3}END{print t "," substr(k, 1, length(k)-2)}' "$MODEL_NAME/results.out")
                run_time=$(awk -F'[/ ]' '/Total time/{print substr($3, 1, length($3)-2)}' "$MODEL_NAME/results.out")
                echo $run_time
                total_time=$(awk -v total="$total_time" -v run="$run_time" 'BEGIN {print total + run}')
        done

        avg_time=$(awk -v total="$total_time" -v runs="$RUNS" 'BEGIN {print total / runs}')
        echo "$test_name,$avg_time" >> "$MODEL_NAME/times"

done

