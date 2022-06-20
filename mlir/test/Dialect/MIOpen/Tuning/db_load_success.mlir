// RUN: miopen-gen -p=false -batchsize=2 -in_channels=64 -out_channels=256 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=0 -padding_w=0 %s | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -debug 2>&1 | FileCheck %s

// CHECK: Successfully opened connection to PerfDb
// CHECK: DB load succeed
