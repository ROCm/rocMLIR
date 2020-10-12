// This tests checks the following aspects of lowering component:
// * gridwise_gemm argument positions are correct

// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

func @miopen_conv2d_cyxk_chwn_khwn(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
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
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 0
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 1
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 2
// CHECK-NEXT:  miopen.gridwise_gemm(%0, %3, %4)

func @miopen_conv2d_bwd_data_cyxk_chwn_khwn(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["c", "y", "x", "k"],
    input_layout = ["ci", "hi", "wi", "ni"],
    output_layout = ["ko", "ho", "wo", "no"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_data
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 0
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 2
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 1
// CHECK-NEXT:  miopen.gridwise_gemm(%0, %4, %3)

func @miopen_conv2d_bwd_weight_cyxk_chwn_khwn(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["c", "y", "x", "k"],
    input_layout = ["ci", "hi", "wi", "ni"],
    output_layout = ["ko", "ho", "wo", "no"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_weight
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 2
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 1
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 0
// CHECK-NEXT:  miopen.gridwise_gemm(%4, %3, %0)
