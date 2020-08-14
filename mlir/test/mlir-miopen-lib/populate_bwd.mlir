// RUN: mlir-miopen-lib-test --args " --operation conv2d_bwd_data --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0" --option cflags | FileCheck %s --check-prefix=CFLAGS
// RUN: mlir-miopen-lib-test --args " --operation conv2d_bwd_data --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0" --option source | FileCheck %s --check-prefix=SOURCE
// RUN: mlir-miopen-lib-test --args " --operation conv2d_bwd_data --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0" --option header | FileCheck %s --check-prefix=HEADER

// CFLAGS: miopen_conv2d_bwd_data_kcyx_nchw_nkhw
// SOURCE: void mlir_gen_igemm_conv2d_v1r1_bwd
// HEADER: struct MlirGenIgemmConv2dV1r1Bwd 
