// RUN: rock-gen -batchsize=1024 -in_channels=1024 -out_channels=1024 -fil_w=1 -fil_h=1 -in_h=14 -in_w=14 -padding_h=1 -padding_w=1 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise | FileCheck %s
// CHECK: leftOobDims = [3 : i32, 4 : i32]
// CHECK-SAME: rightOobDims = [3 : i32, 4 : i32]
