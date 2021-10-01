// RUN: mlir-miopen-e2e-validate.sh \
// RUN: --test-only-ops conv2d conv2d_bwd_weight --end-op-list \
// RUN: --batchsize=256 --groupsize=1 --in_channels=3 --out_channels=64 \
// RUN: --in_h=224 --in_w=224 --fil_h=7 --fil_w=7 --dilation_h=1 \
// RUN: --dilation_w=1 --padding_h=3 --padding_w=3 --conv_stride_h=2 \
// RUN: --conv_stride_w=2

// Test unpadded forward convolutions
// RUN: mlir-miopen-e2e-validate.sh --test-only-ops conv2d --end-op-list \
// RUN: --batchsize=256 --groupsize=1 --in_channels=3 --out_channels=64 \
// RUN: --in_h=230 --in_w=230 --fil_h=7 --fil_w=7 --dilation_h=1 \
// RUN: --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 \
// RUN: --conv_stride_w=2