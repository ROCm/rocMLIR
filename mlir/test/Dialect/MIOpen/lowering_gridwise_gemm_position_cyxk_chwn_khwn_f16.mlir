// This tests checks the following aspects of lowering component:
// * gridwise_gemm argument positions are correct

// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

func @miopen_conv2d_cyxk_chwn_khwn(%filter : memref<8x3x3x128xf16>, %input : memref<8x32x32x128xf16>, %output : memref<128x30x30x128xf16>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["c", "y", "x", "k"],
    input_layout = ["ci", "hi", "wi", "ni"],
    output_layout = ["ko", "ho", "wo", "no"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<8x3x3x128xf16>, memref<8x32x32x128xf16>, memref<128x30x30x128xf16>
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

func @miopen_conv2d_bwd_data_cyxk_chwn_khwn(%filter : memref<8x3x3x128xf16>, %input : memref<8x32x32x128xf16>, %output : memref<128x30x30x128xf16>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["c", "y", "x", "k"],
    input_layout = ["ci", "hi", "wi", "ni"],
    output_layout = ["ko", "ho", "wo", "no"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<8x3x3x128xf16>, memref<8x32x32x128xf16>, memref<128x30x30x128xf16>
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

func @miopen_conv2d_bwd_weight_cyxk_chwn_khwn(%filter : memref<8x3x3x128xf16>, %input : memref<8x32x32x128xf16>, %output : memref<128x30x30x128xf16>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["c", "y", "x", "k"],
    input_layout = ["ci", "hi", "wi", "ni"],
    output_layout = ["ko", "ho", "wo", "no"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<8x3x3x128xf16>, memref<8x32x32x128xf16>, memref<128x30x30x128xf16>
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
