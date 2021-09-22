// -n 256 -c 128 -H 28 -W 28 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --in_channels=128 \
// RUN: --in_h=28 --in_w=28 --out_channels=128 --fil_h=3 --fil_w=3 \
// RUN: --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 \
// RUN: --padding_h=1 --padding_w=1 --groupsize=1

// -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --in_channels=128 \
// RUN: --in_h=28 --in_w=28 --out_channels=512 --fil_h=1 --fil_w=1 \
// RUN: --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 \
// RUN: --padding_h=0 --padding_w=0 --groupsize=1
