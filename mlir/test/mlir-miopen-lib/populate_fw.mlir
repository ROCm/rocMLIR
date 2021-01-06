// RUN: mlir-miopen-lib-test --args " --operation conv2d --arch gfx906 --num_cu 64 --fil_layout NCHW --in_layout NCHW --out_layout NCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0" --option cflags | FileCheck %s --check-prefix=CFLAGS
// RUN: mlir-miopen-lib-test --args " --operation conv2d --arch gfx906 --num_cu 64 --fil_layout NCHW --in_layout NCHW --out_layout NCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0" --option source | FileCheck %s --check-prefix=SOURCE
// RUN: mlir-miopen-lib-test --args " --operation conv2d --arch gfx906 --num_cu 64 --fil_layout NCHW --in_layout NCHW --out_layout NCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0" --option header | FileCheck %s --check-prefix=HEADER

// CFLAGS: miopen_conv2d_kcyx_nchw_nkhw
// SOURCE: void mlir_gen_igemm_conv2d_cpp_v4r4_fwd
// HEADER: struct MlirGenIgemmConv2dV4r4Fwd 
