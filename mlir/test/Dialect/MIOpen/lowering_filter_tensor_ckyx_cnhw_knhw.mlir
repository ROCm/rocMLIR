// This tests checks the following aspects of lowering component:
// * Filter is transformed correctly

// RUN: mlir-opt -miopen-affix-params -miopen-lowering %s | FileCheck %s

func @miopen_conv2d_ckyx_cnhw_knhw(%filter : memref<1x8x128x3x3xf32>, %input : memref<1x8x128x32x32xf32>, %output : memref<1x128x128x30x30xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "c", "k", "y", "x"],
    input_layout = ["gi", "ci", "ni", "hi", "wi"],
    output_layout = ["go", "ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0, 0]
  } : memref<1x8x128x3x3xf32>, memref<1x8x128x32x32xf32>, memref<1x128x128x30x30xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 0
// CHECK:       lower_layer_names = ["g"]
// CHECK:       upper_layer_names = ["gemmG"]
// CHECK:       lower_layer_names = ["c", "y", "x"]
// CHECK:       upper_layer_names = ["gemmK"]
// CHECK:       lower_layer_names = ["k"]
// CHECK:       upper_layer_names = ["gemmM"]
// CHECK-NEXT:  miopen.transform

func @miopen_conv2d_bwd_data_gckyx_gcnhw_gknhw(%filter: memref<1x1024x1024x1x1xf32>, %input: memref<1x1024x128x14x14xf32>, %output: memref<1x1024x128x14x14xf32>) attributes {kernel = 0 : i32} {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx908",
    dilations = [1 : i32, 1 : i32],
    filter_layout = ["g", "c", "k", "y", "x"],
    gemm_id = 0 : i32,
    input_layout = ["gi", "ci", "ni", "hi", "wi"],
    num_cu = 120 : i32,
    output_layout = ["go", "ko", "no", "ho", "wo"],
    padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32],
    strides = [1 : i32, 1 : i32],
    xdlopsV2 = true
  } : memref<1x1024x1024x1x1xf32>, memref<1x1024x128x14x14xf32>, memref<1x1024x128x14x14xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_data
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK:       gridwise_gemm_argument_position = 0
// CHECK:       lower_layer_names = ["g"]
// CHECK:       upper_layer_names = ["gemmG"]
// CHECK:       lower_layer_names = ["k", "ydotslice", "xdotslice"]
// CHECK:       upper_layer_names = ["gemmK"]
// CHECK:       lower_layer_names = ["c", "ytildaslice", "xtildaslice"]
// CHECK:       upper_layer_names = ["gemmM"]
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
    padding = [0, 0, 0, 0]
  } : memref<1x8x128x3x3xf32>, memref<1x8x128x32x32xf32>, memref<1x128x128x30x30xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_weight
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK:       gridwise_gemm_argument_position = 2
// CHECK:       lower_layer_names = ["g"]
// CHECK:       upper_layer_names = ["gemmG"]
// CHECK:       lower_layer_names = ["k"]
// CHECK:       upper_layer_names = ["gemmM"]
// CHECK:       lower_layer_names = ["c", "y", "x"]
// CHECK:       upper_layer_names = ["gemmN"]
// CHECK-NEXT:  miopen.transform
