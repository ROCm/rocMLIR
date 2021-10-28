// This tests checks the following aspects of lowering component:
// * gridwise_gemm argument positions are correct

// RUN: miopen-opt -miopen-affix-params -miopen-lowering %s | FileCheck %s

func @miopen_conv2d_cyxk_chwn_khwn(%filter : memref<1x8x3x3x128xf32>, %input : memref<1x8x32x32x128xf32>, %output : memref<1x128x30x30x128xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "c", "y", "x", "k"],
    input_layout = ["gi", "ci", "hi", "wi", "ni"],
    output_layout = ["go", "ko", "ho", "wo", "no"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0, 0]
  } : memref<1x8x3x3x128xf32>, memref<1x8x32x32x128xf32>, memref<1x128x30x30x128xf32>
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
// CHECK-NEXT:  miopen.gridwise_gemm

func @miopen_conv2d_bwd_data_gcyxk_gcnhw_gknhw(%filter: memref<1x1024x1x1x1024xf32>, %input: memref<1x1024x128x14x14xf32>, %output: memref<1x1024x128x14x14xf32>) attributes {kernel = 0 : i32} {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx908",
    dilations = [1 : i32, 1 : i32],
    filter_layout = ["g", "c", "y", "x", "k"],
    gemm_id = 0 : i32,
    input_layout = ["gi", "ci", "ni", "hi", "wi"],
    num_cu = 120 : i32,
    output_layout = ["go", "ko", "no", "ho", "wo"],
    padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32],
    strides = [1 : i32, 1 : i32],
    xdlopsV2 = true
  } : memref<1x1024x1x1x1024xf32>, memref<1x1024x128x14x14xf32>, memref<1x1024x128x14x14xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_data
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 0
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 2
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 1
// CHECK-NEXT:  miopen.gridwise_gemm

func @miopen_conv2d_bwd_weight_cyxk_chwn_khwn(%filter : memref<1x8x3x3x128xf32>, %input : memref<1x8x32x32x128xf32>, %output : memref<1x128x30x30x128xf32>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "c", "y", "x", "k"],
    input_layout = ["gi", "ci", "hi", "wi", "ni"],
    output_layout = ["go", "ko", "ho", "wo", "no"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0, 0]
  } : memref<1x8x3x3x128xf32>, memref<1x8x32x32x128xf32>, memref<1x128x30x30x128xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_weight
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 2
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 1
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK:       gridwise_gemm_argument_position = 0
// CHECK-NEXT:  miopen.gridwise_gemm
