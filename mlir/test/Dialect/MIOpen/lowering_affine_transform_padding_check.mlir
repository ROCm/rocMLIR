// REQUIRES: miopen-driver
// RUN: mlir-miopen-driver --operation conv2d -t f32 -x2 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 256 --in_channels 128 --in_h 28 --in_w 28 --out_channels 128 --fil_w 3 --fil_h 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -miopen-lowering -miopen-affine-transform -miopen-test-affine-transform | FileCheck %s --check-prefix=CHECK0
// RUN: mlir-miopen-driver --operation conv2d -t f32 -x2 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 256 --in_channels 128 --in_h 28 --in_w 28 --out_channels 128 --fil_w 3 --fil_h 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 -miopen-lowering -miopen-affine-transform -miopen-test-affine-transform | FileCheck %s --check-prefix=CHECK1

// CHECK0: hasPadding = false
// CHECK0-NEXT: hasPadding = false
// CHECK0-NEXT: hasPadding = false
// CHECK0-NEXT: hasPadding = false
// CHECK0-NEXT: hasPadding = false

// CHECK1: hasPadding = false
// CHECK1-NEXT: hasPadding = true
// CHECK1-NEXT: hasPadding = true
// CHECK1-NEXT: hasPadding = true
// CHECK1-NEXT: hasPadding = false
