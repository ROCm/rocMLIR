// RUN: mlir-miopen-driver -p=false -batchsize=2 -in_channels=64 -out_channels=256 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=0 -padding_w=0 -miopen-lowering -miopen-affix-params -debug %s 2>&1 | FileCheck %s --check-prefix=CHECK1
// RUN: mlir-miopen-driver -p=false -batchsize=2 -in_channels=64 -out_channels=256 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=0 -padding_w=0 -t f16 -miopen-lowering -miopen-affix-params -debug %s 2>&1 | FileCheck %s --check-prefix=CHECK2

// CHECK1: Successfully opened connection to PerfDb
// CHECK1: DB load succeed
// CHECK2: Successfully opened connection to PerfDb
// CHECK2: DB load succeed
