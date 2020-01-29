// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

func @miopen_conv2d_ckyx_cnhw_knhw(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    filter_layout = ["c", "k", "y", "x"],
    input_layout = ["ci", "ni", "hi", "wi"],
    output_layout = ["ko", "no", "ho", "wo"],
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
