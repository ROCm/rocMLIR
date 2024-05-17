# #!/bin/bash

 if [ ! -d "gemm/tunedData" ]; then
     mkdir -p "gemm/tunedData"
 fi

../tuningRunner.py --op gemm -d -c ../gemm-configs --output=gemm/tunedData/tuned_CI_gemm.tsv

 counter=0
 while IFS= read -r line
 do
     counter=$((counter + 1))
     python3 ../tuningRunner.py --op gemm --config="$line" -d --output=gemm/tunedData/tuned_gemm_$counter.tsv
 done < "specificGemmConfigs"

 if [ ! -d "conv/tunedData" ]; then
     mkdir -p "conv/tunedData"
 fi

 ../tuningRunner.py --op conv -d -c ../conv-configs --output=conv/tunedData/tuned_CI_conv.tsv

counter=0
while IFS= read -r line
do
    counter=$((counter + 1))
    python3 ../tuningRunner.py --op conv --config="$line" -d --output=conv/tunedData/tuned_conv_$counter.tsv
done <"specificConvConfigs"