// This tests checks the following aspects of lowering component:
// * Filter is transformed correctly

// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

func @miopen_conv2d_ckyx_cnhw_knhw(%filter : memref<1x8x128x3x3xf32>, %input : memref<1x8x128x32x32xf32>, %output : memref<1x128x128x30x30xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "c", "k", "y", "x"],
    input_layout = ["gi", "ci", "ni", "hi", "wi"],
    output_layout = ["go", "ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<1x8x128x3x3xf32>, memref<1x8x128x32x32xf32>, memref<1x128x128x30x30xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 0
// CHECK:       names = ["gemmG"]
// CHECK:       source_names = ["g"]
// CHECK:       names = ["gemmK"]
// CHECK:       source_names = ["c", "y", "x"]
// CHECK:       names = ["gemmM"]
// CHECK:       source_names = ["k"]
// CHECK-NEXT:  miopen.transform

func @miopen_conv2d_bwd_data_ckyx_cnhw_knhw(%filter : memref<1x8x128x3x3xf32>, %input : memref<1x8x128x32x32xf32>, %output : memref<1x128x128x30x30xf32>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "c", "k", "y", "x"],
    input_layout = ["gi", "ci", "ni", "hi", "wi"],
    output_layout = ["go", "ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<1x8x128x3x3xf32>, memref<1x8x128x32x32xf32>, memref<1x128x128x30x30xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_data
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK:       gridwise_gemm_argument_position = 0
// CHECK:       names = ["gemmG"]
// CHECK:       source_names = ["g"]
// CHECK:       names = ["gemmK"]
// CHECK:       source_names = ["k"]
// CHECK:       names = ["gemmM"]
// CHECK:       source_names = ["c", "y", "x"]
// CHECK-NEXT:  miopen.transform(%arg1)

func @miopen_conv2d_bwd_weight_ckyx_cnhw_knhw(%filter : memref<1x8x128x3x3xf32>, %input : memref<1x8x128x32x32xf32>, %output : memref<1x128x128x30x30xf32>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "c", "k", "y", "x"],
    input_layout = ["gi", "ci", "ni", "hi", "wi"],
    output_layout = ["go", "ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<1x8x128x3x3xf32>, memref<1x8x128x32x32xf32>, memref<1x128x128x30x30xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_weight
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK:       gridwise_gemm_argument_position = 2
// CHECK:       names = ["gemmG"]
// CHECK:       source_names = ["g"]
// CHECK:       names = ["gemmM"]
// CHECK:       source_names = ["k"]
// CHECK:       names = ["gemmN"]
// CHECK:       source_names = ["c", "y", "x"]
// CHECK-NEXT:  miopen.transform(%arg1)
