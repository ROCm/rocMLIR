// This tests checks the following aspects of lowering component:
// * transform has the right number of memref
// * gridwise_gemm has the right number of memref

// RUN: mlir-opt -miopen-affix-params -miopen-lowering %s | FileCheck %s

func @miopen_conv2d_kcyx_nchw_nkhw(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni","gi", "ci", "hi", "wi"],
    output_layout = ["no", "go",  "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0, 0]
  } : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.gridwise_gemm.*{.*}.*memref.*memref.*memref}}

func @miopen_conv2d_bwd_data_gkcyx_ngchw_ngkhw(%filter: memref<1x1024x1024x1x1xf32>, %input: memref<128x1x1024x14x14xf32>, %output: memref<128x1x1024x14x14xf32>) attributes {kernel = 0 : i32} {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx908",
    dilations = [1 : i32, 1 : i32],
    filter_layout = ["g", "k", "c", "y", "x"],
    gemm_id = 0 : i32,
    input_layout = ["ni","gi", "ci", "hi", "wi"],
    num_cu = 120 : i32,
    output_layout = ["no", "go",  "ko", "ho", "wo"],
    padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32],
    strides = [1 : i32, 1 : i32],
    xdlopsV2 = true
  } : memref<1x1024x1024x1x1xf32>, memref<128x1x1024x14x14xf32>, memref<128x1x1024x14x14xf32>
  return
}

// CHECK-LABEL: func @miopen_conv2d_bwd_data
// CHECK-NEXT:  {{miopen.transform.*{.*"g", "k", "c", "ydot", "ytilda", "xdot", "xtilda".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"g", "k", "c", "ydotslice", "ytildaslice", "xdotslice", "xtildaslice".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"gemmG", "gemmK", "gemmM".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"gi", "ni", "ci", "hipad", "wipad".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"gi", "ni", "ci", "ytilda", "htilda", "xtilda", "wtilda".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"gi", "ni", "ci", "ytildaslice", "htildaslice", "xtildaslice", "wtildaslice".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"gemmG", "gemmM", "gemmN".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"go", "no", "ko", "ydot", "htilda", "xdot", "wtilda".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"go", "no", "ko", "ydotslice", "htildaslice", "xdotslice", "wtildaslice".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"gemmG", "gemmK", "gemmN".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.gridwise_gemm.*{.*}.*memref.*memref.*memref}}

func @miopen_conv2d_bwd_weight_kcyx_nchw_nkhw(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0, 0]
  } : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_weight
// CHECK-NEXT:  {{miopen.transform.*{.*"g", "k", "c", "y", "x".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"gemmG", "gemmM", "gemmNPad".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"ni", "gi", "ci", "hipad", "wipad".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"ni", "gi", "ci", "y", "ho", "x", "wo".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"gemmG", "gemmK", "gemmN".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"gemmG", "gemmK", "gemmNPad".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*"gemmG", "gemmK", "gemmM".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.gridwise_gemm.*{.*}.*memref.*memref.*memref}}
