#!/bin/bash
 
# Shell script that captures the performance difference between data types fp16 and int8 to validate expected kernel performance.
# The script is currently comparing only fp16 vs int8 performance of non-fused kernels (only convolution).
# Usage: ./performance-checking.sh --d <model> --p <model_path> [--r <number_of_iterations>]
 
MODEL_NAME="resnet50-fp16"
MODEL_PATH="/mnt/sc_nas_share/migraphx/models/mlperf/resnet50_v1.onnx"
RUNS=5
 
export PATH="$HOME/AMDMIGraphX/build/bin:$PATH" # path to migraphx-driver
 
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
 
mkdir -p "$MODEL_NAME"
 
# Compile model to generate .mxr files
MIGRAPHX_MLIR_DUMP_TO_MXR="$MODEL_NAME" \
MIGRAPHX_ENABLE_NHWC=1 \
MIGRAPHX_ENABLE_HIPBLASLT_GEMM=1 \
MIGRAPHX_MLIR_USE_SPECIFIC_OPS="convolution,~fused,~dot" \
migraphx-driver compile "$MODEL_PATH" --fp16 --exhaustive-tune
 
# Convert each .mxr to .py
ls "$MODEL_NAME"/*.mxr | xargs -I {} -n 1 migraphx-driver read "{}" --py -o "{}".py
 
# Benchmark a set of .py testcases
run_benchmark() {
    local label="$1"
    echo "$label:" >> "$MODEL_NAME/times"
 
    for testcase in "$MODEL_NAME"/*.py; do
        test_name=$(basename "$testcase")
        total_time=0
 
        compiled="$testcase.mxdb"
        migraphx-driver compile "$testcase" --mlir -o "$compiled" > /dev/null
 
        for ((i = 1; i <= RUNS; i++)); do
            migraphx-driver time "$compiled" > "$MODEL_NAME/results.out"
            run_time=$(awk -F'[/ ]' '/Total time/{print substr($3, 1, length($3)-2)}' "$MODEL_NAME/results.out")
            echo "$run_time"
            total_time=$(awk -v total="$total_time" -v run="$run_time" 'BEGIN {print total + run}')
        done
 
        avg_time=$(awk -v total="$total_time" -v runs="$RUNS" 'BEGIN {print total / runs}')
        echo "$test_name,$avg_time" >> "$MODEL_NAME/times"
    done
}
 
echo "NEW RUN" >> "$MODEL_NAME/times"
run_benchmark "FP16"
 
# Modify the testcases for int8 quantization
sed -i -e "s/half_type/int8_type/" -e 's/convolution/quant_convolution/' "$MODEL_NAME"/*.py
 
run_benchmark "INT8"
