#!/bin/bash

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
        *)
                        echo "Option $1 doesn't exist"
                        exit 1
                        ;;
        esac
done

#if [ -z "$MODEL_NAME" ]; then 
#       echo "Provide model: $0 --m <model_name>"
#       exit 1
#fi

mkdir "$MODEL_NAME"

MIGRAPHX_MLIR_DUMP_TO_MXR="$MODEL_NAME" MIGRAPHX_ENABLE_NHWC=1 MIGRAPHX_ENABLE_HIPBLASLT_GEMM=1 MIGRAPHX_MLIR_USE_SPECIFIC_OPS="convolution,~fused,~dot" migraphx-driver compile "$MODEL_PATH" --fp16 --exhaustive-tune

ls "$MODEL_NAME"/*.mxr |xargs -I {} -n 1 migraphx-driver read "{}" --py -o "{}".py

sed -i -e 's/half_type/int8_type/' -e 's/convolution/quant_convolution/' "$MODEL_NAME"/*.py

echo "NEW RUN" >> "$MODEL_NAME/times"

#while read testcase
#do
#       MIGRAPHX_DISABLE_PASSES=auto_contiguous migraphx-driver time $testcase --mlir > results.out
#       awk -F'[/ ]' -v t="$testcase" '/Total time/{k=$3}END{print t "," substr(k, 1, length(k)-2)}' results.out >> times
#done <<TESTLIST

for testcase in "$MODEL_NAME"/*.py
do
        test_name=$(basename "$testcase")
#        MIGRAPHX_DISABLE_PASSES=auto_contiguous migraphx-driver time $testcase --mlir > "$MODEL_NAME/results.out"
#        awk -F'[/ ]' -v t="$test_name" '/Total time/{k=$3}END{print t "," substr(k, 1, length(k)-2)}' "$MODEL_NAME/results.out" >> "$MODEL_NAME/times"
#done

#test_name=$(basename "$testcase")
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


# <<TESTLIST
#$(find "$MODEL_NAME" -type f -name "*.py")
#TESTLIST
