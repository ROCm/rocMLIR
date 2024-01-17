#!/bin/bash
# This is the script that is executed for MigraphX Integration CI and serves for testing models.

set -e

# We have two testing options. When checkFor is set to perf, it will initiate performance measurements for the models.
# On the other hand, if we want to run model verification, we should replace perf with verify.
checkFor="perf"

ARCH=$(rocminfo |grep -o -m 1 'gfx.*' | tr -d '[:space:]')
echo -n "Architecture: $ARCH"
echo "$ARCH" > /logs/arch.txt

SUMMARY="/logs/${ARCH}_summary.log"
LOGFILE="/logs/${ARCH}_generic.log"

rm -f $LOGFILE
rm -f $SUMMARY

echo "###########################################" >  $LOGFILE
echo "New Run $(pwd)" >>  $LOGFILE
date >> $LOGFILE
echo "GPU: $(rocminfo |grep -o -m 1 'gfx.*')" >> $LOGFILE
echo "MIGX: $(/AMDMIGraphX/build/bin/migraphx-driver --version)" >> $LOGFILE
echo "MIGX Commit: $(git -C /AMDMIGraphX log -n 1  --pretty=oneline)" >> $LOGFILE
ls -l /etc/alternatives |grep "rocm ->" >> $LOGFILE
echo "###########################################" >>  $LOGFILE

# If we want to disable quantization for fp16 or int8, we need to change it from true to false.
fp32="true"
fp16="true"
int8="true"

datatypes=()
if [ "$fp32" = "true" ]; then
        datatypes+=(" ")
fi
if [ "$fp16" = "true" ]; then
        datatypes+=("--fp16")
fi
if [ "$int8" = "true" ]; then
        datatypes+=("--int8")
fi

echo "Data type flags:"
printf -- '- %s\n' "${datatypes[@]}"

# Create lists of models which we want to perform testing. If we want to add or remove models we can change lists of models.
list_tier1_p0="/models/ORT/bert_base_cased_1.onnx --fill1 input_ids --input-dim @input_ids 1 384 --batch 1
/models/ORT/bert_base_cased_1.onnx --fill1 input_ids --input-dim @input_ids 32 384 --batch 32
/models/ORT/bert_base_cased_1.onnx --fill1 input_ids --input-dim @input_ids 64 384 --batch 64
/models/ORT/bert_base_uncased_1.onnx --fill1 input_ids --input-dim @input_ids 1 384 --batch 1
/models/ORT/bert_base_uncased_1.onnx --fill1 input_ids --input-dim @input_ids 32 384 --batch 32
/models/ORT/bert_base_uncased_1.onnx --fill1 input_ids --input-dim @input_ids 64 384 --batch 64
/models/ORT/bert_large_uncased_1.onnx --fill1 input_ids --input-dim @input_ids 1 384 --batch 64
/models/ORT/distilgpt2_1.onnx --fill1 input_ids --input-dim @input_ids 1 384 --batch 1
/models/ORT/distilgpt2_1.onnx --fill1 input_ids --input-dim @input_ids 32 384 --batch 32
/models/ORT/distilgpt2_1.onnx --fill1 input_ids --input-dim @input_ids 64 384 --batch 64
/models/ORT/onnx_models/bert_base_cased_1_fp16_gpu.onnx --fill1 input_ids --input-dim @input_ids 1 384 --batch 1
/models/ORT/onnx_models/bert_base_cased_1_fp16_gpu.onnx --fill1 input_ids --input-dim @input_ids 32 384 --batch 32
/models/ORT/onnx_models/bert_base_cased_1_fp16_gpu.onnx --fill1 input_ids --input-dim @input_ids 64 384 --batch 64
/models/ORT/onnx_models/bert_large_uncased_1_fp16_gpu.onnx --fill1 input_ids --input-dim @input_ids 1 384 --batch 1
/models/ORT/onnx_models/bert_large_uncased_1_fp16_gpu.onnx --fill1 input_ids --input-dim @input_ids 32 384 --batch 32
/models/ORT/onnx_models/bert_large_uncased_1_fp16_gpu.onnx --fill1 input_ids --input-dim @input_ids 64 384 --batch 64
/models/ORT/onnx_models/distilgpt2_1_fp16_gpu.onnx --fill1 input_ids --input-dim @input_ids 1 384 --batch 1
/models/ORT/onnx_models/distilgpt2_1_fp16_gpu.onnx --fill1 input_ids --input-dim @input_ids 32 384 --batch 32
/models/ORT/onnx_models/distilgpt2_1_fp16_gpu.onnx --fill1 input_ids --input-dim @input_ids 64 384 --batch 64
/models/onnx-model-zoo/gpt2-10.onnx
/models/mlperf/resnet50_v1.onnx"

list_tier1_p1="/models/sd/stable-diffusion-2-onnx/text_encoder/model.onnx --input-dim @latent_sample 1 4 64 64 -t 482
/models/sd/stable-diffusion-2-onnx/vae_decoder/model.onnx --input-dim @latent_sample 1 4 64 64 -t 482
/models/mlperf/bert_large_mlperf.onnx --fill1 input_ids --fill1 input_ids --fill1 segment_ids --input-dim @input_ids 1 384
/models/mlperf/bert_large_mlperf.onnx --fill1 input_ids --fill1 input_ids --fill1 segment_ids --input-dim @input_ids 64 384
/models/sd/stable-diffusion-2-onnx/unet/model.onnx --input-dim @sample 2 4 64 64 @timestep 1 @encoder_hidden_states 2 64 1024"

list_others=""

echo "Collecting models:"
echo -e "$list_tier1_p0"
echo -e "$list_tier1_p1"
echo -e "$list_others"

tier1_p0_models=()
tier1_p1_models=()
other_models=()

while IFS= read -r line; do
    tier1_p0_models+=("$line")
done <<< "$list_tier1_p0"
while IFS= read -r line; do
    tier1_p1_models+=("$line")
done <<< "$list_tier1_p1"
while IFS= read -r line; do
    other_models+=("$line")
done <<< "$list_others"

# Function to test different list of models
function test_models(){
  array_name=$1[@]
  models_to_test=("${!array_name}")
  out_log_file=$2
  for testcase in "${models_to_test[@]}"; do
      if [[ $str =~ ^# ]]; then
          continue;
      fi
      for datatype in "${datatypes[@]}"; do
          echo "Testing: $testcase $datatype" >> $out_log_file
          timeout 1h env MIGRAPHX_ENABLE_MLIR=1 /AMDMIGraphX/build/bin/migraphx-driver $checkFor $testcase $datatype 2>&1 |tee raw_log.txt
          timeout_status=$?
          cat raw_log.txt |sed -n '/Summary:/,$p'  >>  $out_log_file
          cat raw_log.txt |sed -n '/FAILED:/,$p'  >>  $out_log_file
          result="DONE"
          if [[ $timeout_status -eq 124 ]]; then
                  result="TIMEOUT"
          fi
          echo "$result Testing(MLIR ENABLED): $testcase $datatype" >> $out_log_file
          echo "(MLIR ENABLED) $testcase $datatype $result" >> $SUMMARY
      done
  done
}
rm -f tier1_p0.log
rm -f tier1_p1.log
rm -f other_models.log

# Enable tests for different models group.
enable_tier1_p0="true"
enable_tier1_p1="true"
enable_others="false"

if [ "$enable_tier1_p0" = "true" ]; then
    test_models tier1_p0_models /logs/${ARCH}_tier1_p0.log
fi
if [ "$enable_tier1_p1" = "true" ]; then
    test_models tier1_p1_models /logs/${ARCH}_tier1_p1.log
fi
if [ "$enable_others" = "true" ]; then
    test_models other_models /logs/${ARCH}_other_models.log
fi 
