#!/bin/bash

set -e

checkFor="verify"

ARCH=$(rocminfo |grep -o -m 1 'gfx.*' | tr -d '[:space:]')  
echo -n "Architecture: $ARCH"
echo "$ARCH" > /logs/arch.txt 

SUMMARY="/logs/${ARCH}_summary.log"
LOGFILE="/logs/${ARCH}_generic.log"

echo "LOGFILE: $LOGFILE"  
ls -ld /logs  
ls -l $LOGFILE || true 

rm -f $LOGFILE
rm -f $SUMMARY

echo "###########################################" >  $LOGFILE
echo "New Run $(pwd)" >  $LOGFILE
date > $LOGFILE
echo "GPU: $(rocminfo |grep -o -m 1 'gfx.*')" > $LOGFILE
echo "MIGX: $(/AMDMIGraphX/build/bin/migraphx-driver --version)" > $LOGFILE
echo "MIGX Commit: $(git -C /AMDMIGraphX log -n 1  --pretty=oneline)" > $LOGFILE
ls -l /etc/alternatives |grep "rocm ->" > $LOGFILE
echo "###########################################" >  $LOGFILE

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

list_tier1_p0="/models/ORT/bert_base_cased_1.onnx --fill1 input_ids --input-dim @input_ids 1 384 --batch 1
/models/ORT/bert_base_cased_1.onnx --fill1 input_ids --input-dim @input_ids 32 384 --batch 32"

list_tier1_p1="/models/sd/stable-diffusion-2-onnx/text_encoder/model.onnx --input-dim @latent_sample 1 4 64 64 -t 482
/models/sd/stable-diffusion-2-onnx/vae_decoder/model.onnx --input-dim @latent_sample 1 4 64 64 -t 482"

echo "Collecting models:"
echo -e "$list_tier1_p0"
echo -e "$list_tier1_p1"

tier1_p0_models=()
tier1_p1_models=()

while IFS= read -r line; do  
    tier1_p0_models+=("$line")  
done <<< "$list_tier1_p0"  
while IFS= read -r line; do  
    tier1_p1_models+=("$line")  
done <<< "$list_tier1_p1"  

#echo "Before read: IFS = '$IFS'" 
#IFS=$'\n' read -r -a tier1_p0_models <<< "$list_tier1_p0"
#echo "After read: tier1_p0_models = ${tier1_p0_models[@]}"
#IFS=$'\n' read -r -a tier1_p1_models <<< "$list_tier1_p1"

other_models=()
# Ispisivanje modela za proveru  
for model in "${tier1_p0_models[@]}"; do  
    echo "Model: $model"  
done  
  
for model in "${tier1_p1_models[@]}"; do  
    echo "Model: $model"  
done  

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
         
          #echo "Testing(MLIR ENABLED): \$testcase \$datatype" >> \$out_log_file
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
          echo "kraj"       
      done
  done
}
rm -f tier1_p0.log
rm -f tier1_p1.log
rm -f other_models.log

enable_tier1_p0="true"
enable_tier1_p1="true"

if [ "$enable_tier1_p0" = "true" ]; then
	test_models tier1_p0_models /logs/${ARCH}_tier1_p0.log
fi
if [ "$enable_tier1_p1" = "true" ]; then
    test_models tier1_p1_models /logs/${ARCH}_tier1_p1.log
fi
if [ "$enable_others" = "true" ]; then
    test_models other_models /logs/${ARCH}_other_models.log
fi 
echo "LOGFILE"
cat $LOGFILE
echo "SUMMARY"
cat $SUMMARY
echo "tier1_p0.log"
cat /logs/${ARCH}_tier1_p0.log
echo "tier1_p1.log"
cat /logs/${ARCH}_tier1_p1.log
#ls -l $LOGFILE  
#ls -l $SUMMARY  
#ls -l $(pwd)/tier1_p0.log  
#ls -l $(pwd)/tier1_p1.log  