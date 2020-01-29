// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

func @miopen_conv2d_cyxk_chwn_khwn(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    filter_layout = ["c", "y", "x", "k"],
    input_layout = ["ci", "hi", "wi", "ni"],
    output_layout = ["ko", "ho", "wo", "no"],
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
