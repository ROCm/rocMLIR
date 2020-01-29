// RUN: mlir-opt -miopen-lowering -split-input-file %s | FileCheck %s

func @miopen_conv2d_kyxc_nhwc_nhwk(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    filter_layout = ["k", "y", "x", "c"],
    input_layout = ["ni", "hi", "wi", "ci"],
    output_layout = ["no", "ho", "wo", "ko"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d
//  CHECK-NOT: miopen.conv2d
//  CHECK-NEXT: miopen.transform
//  CHECK-NEXT: miopen.transform
//  CHECK-NEXT: miopen.transform
//  CHECK-NEXT: miopen.transform
//  CHECK-NEXT: miopen.transform
//  CHECK-NEXT: miopen.gridwise_gemm
