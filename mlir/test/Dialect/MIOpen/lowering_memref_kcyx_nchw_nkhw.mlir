// This tests checks the following aspects of lowering component:
// * transform has the right number of memref
// * gridwise_gemm has the right number of memref

// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

func @miopen_conv2d_kcyx_nchw_nkhw(%filter : memref<1x128x8x3x3xf32>, %input : memref<1x128x8x32x32xf32>, %output : memref<1x128x128x30x30xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["gi", "ni", "ci", "hi", "wi"],
    output_layout = ["go", "no", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<1x128x8x3x3xf32>, memref<1x128x8x32x32xf32>, memref<1x128x128x30x30xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.gridwise_gemm.*{.*}.*memref.*memref.*memref}}

func @miopen_conv2d_bwd_data_kcyx_nchw_nkhw(%filter : memref<1x128x8x3x3xf32>, %input : memref<1x128x8x32x32xf32>, %output : memref<1x128x128x30x30xf32>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["gi", "ni", "ci", "hi", "wi"],
    output_layout = ["go", "no", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<1x128x8x3x3xf32>, memref<1x128x8x32x32xf32>, memref<1x128x128x30x30xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_data
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.gridwise_gemm.*{.*}.*memref.*memref.*memref}}

func @miopen_conv2d_bwd_weight_kcyx_nchw_nkhw(%filter : memref<1x128x8x3x3xf32>, %input : memref<1x128x8x32x32xf32>, %output : memref<1x128x128x30x30xf32>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["gi", "ni", "ci", "hi", "wi"],
    output_layout = ["go", "no", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<1x128x8x3x3xf32>, memref<1x128x8x32x32xf32>, memref<1x128x128x30x30xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_weight
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.gridwise_gemm.*{.*}.*memref.*memref.*memref}}
