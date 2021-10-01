// Configs 9-16
// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --groupsize=1 \
// RUN: --in_channels=256 --in_h=14 --in_w=14 --out_channels=256 \
// RUN: --fil_h=3 --fil_w=3 --dilation_h=1 --dilation_w=1 \
// RUN: --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1

// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --groupsize=1 \
// RUN: --in_channels=256 --in_h=30 --in_w=30 --out_channels=256 \
// RUN: --fil_h=3 --fil_w=3 --dilation_h=1 --dilation_w=1 \
// RUN: --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0

// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --groupsize=1 \
// RUN: --in_channels=256 --in_h=56 --in_w=56 --out_channels=128 \
// RUN: --fil_h=1 --fil_w=1 --dilation_h=1 --dilation_w=1 \
// RUN: --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0

// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --groupsize=1 \
// RUN: --in_channels=256 --in_h=56 --in_w=56 --out_channels=512 \
// RUN: --fil_h=1 --fil_w=1 --dilation_h=1 --dilation_w=1 \
// RUN: --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0

// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --groupsize=1 \
// RUN: --in_channels=256 --in_h=56 --in_w=56 --out_channels=64 \
// RUN: --fil_h=1 --fil_w=1 --dilation_h=1 --dilation_w=1 \
// RUN: --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0

// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --groupsize=1 \
// RUN: --in_channels=512 --in_h=16 --in_w=16 --out_channels=512 \
// RUN: --fil_h=3 --fil_w=3 --dilation_h=1 --dilation_w=1 \
// RUN: --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0

// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --groupsize=1 \
// RUN: --in_channels=512 --in_h=28 --in_w=28 --out_channels=1024 \
// RUN: --fil_h=1 --fil_w=1 --dilation_h=1 --dilation_w=1 \
// RUN: --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0

// RUN: mlir-miopen-e2e-validate.sh --batchsize=256 --groupsize=1 \
// RUN: --in_channels=512 --in_h=28 --in_w=28 --out_channels=128 \
// RUN: --fil_h=1 --fil_w=1 --dilation_h=1 --dilation_w=1 \
// RUN: --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0
