// This tests checks the following aspects of lowering component:
// * transform has the right number of memref
// * gridwise_gemm has the right number of memref

// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

func @miopen_conv2d_kcyx_nchw_nkhw_padding_kernel(%filter : memref<32x4x2x3x3xf32>, %input : memref<2x32x2x11x11xf32>, %output : memref<2x32x4x9x9xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni","gi", "ci", "hi", "wi"],
    output_layout = ["no", "go",  "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<32x4x2x3x3xf32>, memref<2x32x2x11x11xf32>, memref<2x32x4x9x9xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d
// CHECK-NEXT:  {{miopen.transform.*{extraPad = "true", gemmK_extra = 14 : i32, gemmM_extra = 60 : i32,.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{extraPad = "true", gemmK_extra = 14 : i32, gemmN_extra = 30 : i32,.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{extraPad = "true", gemmM_extra = 60 : i32, gemmN_extra = 30 : i32,.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.gridwise_gemm.*{.*}.*memref.*memref.*memref}}

func @miopen_conv2d_kcyx_nchw_nkhw_no_extra_padding(%filter : memref<1x128x64x3x3xf32>, %input : memref<128x1x64x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni","gi", "ci", "hi", "wi"],
    output_layout = ["no", "go",  "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<1x128x64x3x3xf32>, memref<128x1x64x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d
// CHECK-NEXT:  {{miopen.transform.*{extraPad = "false", gemmK_extra = 0 : i32, gemmM_extra = 0 : i32,.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{extraPad = "false", gemmK_extra = 0 : i32, gemmN_extra = 0 : i32,.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{extraPad = "false", gemmM_extra = 0 : i32, gemmN_extra = 0 : i32,.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.gridwise_gemm.*{.*}.*memref.*memref.*memref}}

func @miopen_conv2d_kcyx_nchw_nkhw_partial_padding_kernel(%filter : memref<32x4x2x3x3xf32>, %input : memref<128x32x2x11x11xf32>, %output : memref<128x32x4x9x9xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni","gi", "ci", "hi", "wi"],
    output_layout = ["no", "go",  "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<32x4x2x3x3xf32>, memref<128x32x2x11x11xf32>, memref<128x32x4x9x9xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d
// CHECK-NEXT:  {{miopen.transform.*{extraPad = "true", gemmK_extra = 14 : i32, gemmM_extra = 60 : i32,.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{extraPad = "true", gemmK_extra = 14 : i32, gemmN_extra = 0 : i32,.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{extraPad = "true", gemmM_extra = 60 : i32, gemmN_extra = 0 : i32,.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.gridwise_gemm.*{.*}.*memref.*memref.*memref}}


