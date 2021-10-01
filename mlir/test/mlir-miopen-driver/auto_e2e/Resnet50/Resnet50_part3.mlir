// Configs 17-22
// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --groupsize=1 \
// RUN: --in_channels=512 --in_h=28 --in_w=28 --out_channels=256 \
// RUN: --fil_h=1 --fil_w=1 --dilation_h=1 --dilation_w=1 \
// RUN: --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0

// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 -groupsize=1 \
// RUN: --in_channels=512 --in_h=7 --in_w=7 --out_channels=2048 \
// RUN: --fil_h=1 --fil_w=1 --dilation_h=1 --dilation_w=1 \
// RUN: --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0

// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --groupsize=1 \
// RUN: --in_channels=512 --in_h=7 --in_w=7 --out_channels=512 \
// RUN: --fil_h=3 --fil_w=3 --dilation_h=1 --dilation_w=1 \
// RUN: --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1

// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --groupsize=1 \
// RUN: --in_channels=64 --in_h=56 --in_w=56 --out_channels=256 \
// RUN: --fil_h=1 --fil_w=1 --dilation_h=1 --dilation_w=1 \
// RUN: --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0

// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --groupsize=1 \
// RUN: --in_channels=64 --in_h=56 --in_w=56 --out_channels=64 \
// RUN: --fil_h=1 --fil_w=1 --dilation_h=1 --dilation_w=1 \
// RUN: --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0

// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --groupsize=1 \
// RUN: --in_channels=64 --in_h=56 --in_w=56 --out_channels=64 \
// RUN: --fil_h=3 --fil_w=3 --dilation_h=1 --dilation_w=1 \
// RUN: --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1
