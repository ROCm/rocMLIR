// RUN: mlir-miopen-e2e-validate.sh \
// RUN: --test-only-ops conv2d conv2d_bwd_weight --end-op-list \
// RUN: --batchsize=20 --groupsize=1 --in_channels=3 --out_channels=6 \
// RUN: --in_h=32 --in_w=32 --fil_h=7 --fil_w=7 --dilation_h=1 --dilation_w=1 \
// RUN: --padding_h=3 --padding_w=3 --conv_stride_h=2 --conv_stride_w=2