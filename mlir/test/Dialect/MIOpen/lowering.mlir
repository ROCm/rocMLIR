// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

func @miopen_conv2d(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    filter_layout = ["k", "c", "y", "x"],
    input_layout = ["ni", "ci", "hi", "wi"],
    output_layout = ["no", "ko", "ho", "wo"],
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
