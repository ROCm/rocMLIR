// XFAIL: *
// RUN: mlir-rock-lib-test --option=bin --args "--operation conv2d \
// RUN: --arch amdgcn-amd-amdhsa:gfx908 --num_cu 64 --in_type fp32 \
// RUN: --fil_type fp32 --out_type fp32  --in_layout NGCHW --fil_layout GNCHW \
// RUN: --out_layout NGCHW --batchsize 128 --in_channels 256 --in_h 256 \
// RUN: --in_w 256 --out_channels 4 --fil_w 3 --fil_h 3 --padding_h 0 \
// RUN: --padding_w 0 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 \
// RUN: --conv_stride_w 1 --out_h 254 --out_w 254 --kernel_name big \
// RUN: --groupsize 1"
