// This tests checks the following aspects of lowering component:
// * Filter is transformed correctly

// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

func @miopen_conv2d_ckyx_cnhw_knhw(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
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
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 0
// CHECK:       names = ["gemmK"]
// CHECK:       source_names = ["c", "y", "x"]
// CHECK:       names = ["gemmM"]
// CHECK:       source_names = ["k"]
// CHECK-NEXT:  miopen.transform

func @miopen_conv2d_bwd_data_ckyx_cnhw_knhw(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["c", "k", "y", "x"],
    input_layout = ["ci", "ni", "hi", "wi"],
    output_layout = ["ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_data
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK:       gridwise_gemm_argument_position = 0
// CHECK:       names = ["gemmK"]
// CHECK:       source_names = ["k"]
// CHECK:       names = ["gemmM"]
// CHECK:       source_names = ["c", "y", "x"]
// CHECK-NEXT:  miopen.transform(%arg1)

func @miopen_conv2d_bwd_weight_ckyx_cnhw_knhw(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["c", "k", "y", "x"],
    input_layout = ["ci", "ni", "hi", "wi"],
    output_layout = ["ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_weight
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK:       gridwise_gemm_argument_position = 2
// CHECK:       names = ["gemmM"]
// CHECK:       source_names = ["k"]
// CHECK:       names = ["gemmN"]
// CHECK:       source_names = ["c", "y", "x"]
// CHECK-NEXT:  miopen.transform(%arg1)
