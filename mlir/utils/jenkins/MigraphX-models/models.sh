#!/bin/bash

SUMMARY=summary.log
LOGFILE=generic.log
rm -f \$LOGFILE
rm -f \$SUMMARY

echo "###########################################" >>  \$LOGFILE
echo "New Run \$(pwd)" >>  \$LOGFILE
date >> \$LOGFILE
echo "GPU: \$(rocminfo |grep -o -m 1 'gfx.*')" >> \$LOGFILE
echo "MIGX: \$(/AMDMIGraphX/build/bin/migraphx-driver --version)" >> \$LOGFILE
echo "MIGX Commit: \$(git -C /AMDMIGraphX log -n 1  --pretty=oneline)" >> \$LOGFILE
ls -l /etc/alternatives |grep "rocm ->" >> \$LOGFILE
echo "###########################################" >>  \$LOGFILE

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
printf -- '- %s\n' "\${datatypes[@]}"
echo "$list_tier1_p0"
echo "Collecting models:"
tier1_p0_models=()
tier1_p1_models=()
other_models=()
OLD_IFS="\$IFS"
IFS=\$'\n'
for model in "$list_tier1_p0"; do
tier1_p0_models+=(\$model)
done
for model in "$list_tier1_p1"; do
tier1_p1_models+=(\$model)
done
for model in "$list_others"; do
other_models+=(\$model)
done
IFS="\$OLD_IFS"


# Function to test different list of models
function test_models(){
  array_name=\$1[@]
  models_to_test=("\${!array_name}")
  out_log_file=\$2
  for testcase in "\${models_to_test[@]}"; do
      if [[ \$str =~ ^# ]]; then
          continue;
      fi
      
      for datatype in "\${datatypes[@]}"; do
          echo "Testing: \$testcase \$datatype" >> \$out_log_file
          timeout 1h docker exec -e MIGRAPHX_ENABLE_MLIR=0  migraphx /AMDMIGraphX/build/bin/migraphx-driver $checkFor \$testcase \$datatype 2>&1 |tee raw_log.txt
          timeout_status=\$?
          cat raw_log.txt |sed -n '/Summary:/,\$p'  >>  \$out_log_file
          cat raw_log.txt |sed -n '/FAILED:/,\$p'  >>  \$out_log_file
          result="DONE"
          if [[ \$timeout_status -eq 124 ]]; then
                  result="TIMEOUT"
          fi
          echo "\$result Testing: \$testcase \$datatype" >> \$out_log_file
          echo "\$testcase \$datatype \$result" >> \$SUMMARY
          
          echo "Testing(MLIR ENABLED): \$testcase \$datatype" >> \$out_log_file
          timeout 1h docker exec -e MIGRAPHX_ENABLE_MLIR=1 migraphx /AMDMIGraphX/build/bin/migraphx-driver \$checkFor \$testcase \$datatype 2>&1 |tee raw_log.txt
          timeout_status=\$?
          cat raw_log.txt |sed -n '/Summary:/,\$p'  >>  \$out_log_file
          cat raw_log.txt |sed -n '/FAILED:/,\$p'  >>  \$out_log_file
          result="DONE"
          if [[ \$timeout_status -eq 124 ]]; then
                  result="TIMEOUT"
          fi
          echo "\$result Testing(MLIR ENABLED): \$testcase \$datatype" >> \$out_log_file
          echo "(MLIR ENABLED) \$testcase \$datatype \$result" >> \$SUMMARY       
      done
  done
}
rm -f tier1_p0.log
rm -f tier1_p1.log
rm -f other_models.log

if [ "$enable_tier1_p0" = "true" ]; then
	test_models tier1_p0_models tier1_p0.log
fi
if [ "$enable_tier1_p1" = "true" ]; then
    test_models tier1_p1_models tier1_p1.log
fi

if [ "$enable_others" = "true" ]; then
    test_models other_models other_models.log
fi 
