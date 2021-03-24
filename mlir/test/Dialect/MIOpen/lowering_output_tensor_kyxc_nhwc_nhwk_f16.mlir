// This tests checks the following aspects of lowering component:
// * Has the correct attribute to output tensor

// RUN: mlir-opt -miopen-lowering -split-input-file %s | FileCheck %s

func @miopen_conv2d_kyxc_nhwc_nhwk(%filter : memref<128x3x3x8xf16>, %input : memref<128x32x32x8xf16>, %output : memref<128x30x30x128xf16>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["k", "y", "x", "c"],
    input_layout = ["ni", "hi", "wi", "ci"],
    output_layout = ["no", "ho", "wo", "ko"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<128x3x3x8xf16>, memref<128x32x32x8xf16>, memref<128x30x30x128xf16>
  return
}
// CHECK-LABEL: func @miopen_conv2d
// CHECK:       miopen.transform(%arg2)
// CHECK:       names = ["gemmM"]
// CHECK:       source_names = ["ko"]
// CHECK:       names = ["gemmN"]
// CHECK:       source_names = ["no", "ho", "wo"]
// CHECK:       miopen.gridwise_gemm

func @miopen_conv2d_bwd_data_kyxc_nhwc_nhwk(%filter : memref<128x3x3x8xf16>, %input : memref<128x32x32x8xf16>, %output : memref<128x30x30x128xf16>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["k", "y", "x", "c"],
    input_layout = ["ni", "hi", "wi", "ci"],
    output_layout = ["no", "ho", "wo", "ko"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<128x3x3x8xf16>, memref<128x32x32x8xf16>, memref<128x30x30x128xf16>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_data
// CHECK:       miopen.transform(%arg2)
// CHECK:       names = ["gemmK"]
// CHECK:       source_names = ["ko"]
// CHECK:       names = ["gemmN"]
// CHECK:       source_names = ["no", "ho", "wo"]
// CHECK:       miopen.gridwise_gemm

func @miopen_conv2d_bwd_weight_kyxc_nhwc_nhwk(%filter : memref<128x3x3x8xf16>, %input : memref<128x32x32x8xf16>, %output : memref<128x30x30x128xf16>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["k", "y", "x", "c"],
    input_layout = ["ni", "hi", "wi", "ci"],
    output_layout = ["no", "ho", "wo", "ko"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<128x3x3x8xf16>, memref<128x32x32x8xf16>, memref<128x30x30x128xf16>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_weight
// CHECK:       miopen.transform(%arg2)
// CHECK:       names = ["gemmK"]
// CHECK:       source_names = ["no", "ho", "wo"]
// CHECK:       names = ["gemmM"]
// CHECK:       source_names = ["ko"]
// CHECK:       miopen.gridwise_gemm
