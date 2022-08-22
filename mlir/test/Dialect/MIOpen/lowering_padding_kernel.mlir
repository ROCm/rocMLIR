// This tests checks the following aspects of lowering component:
// * The correct padding transformations are generated and added to the gemm

// RUN: miopen-opt -miopen-affix-params -miopen-conv-to-gemm %s | FileCheck %s
// CHECK-DAG: #[[$PAD_GEMMK:.*]] = #miopen.transform_map{{.*}}Pad{0, 14} ["gemmKPad"] at [1] -> ["gemmK"] at [1]

// CHECK-LABEL: func.func @miopen_conv2d_kcyx_nchw_nkhw_padding_kernel
// CHECK-SAME: %[[filter:.*]]: memref<32x128x2x3x3xf32>
// CHECK: %[[gemmFilter:.*]] = miopen.transform %[[filter]]
// CHECK: %[[padK:.*]] = miopen.transform %[[gemmFilter]] by [#[[$PAD_GEMMK]]]
// CHECK: miopen.gridwise_gemm(%[[padK]], %{{.*}}, %{{.*}})
func.func @miopen_conv2d_kcyx_nchw_nkhw_padding_kernel(%filter : memref<32x128x2x3x3xf32>, %input : memref<64x32x2x11x11xf32>, %output : memref<64x32x128x9x9xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    numCu = 64 : i32,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni","gi", "ci", "hi", "wi"],
    output_layout = ["no", "go",  "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<32x128x2x3x3xf32>, memref<64x32x2x11x11xf32>, memref<64x32x128x9x9xf32>
  return
}

// CHECK-LABEL: func.func @miopen_conv2d_kcyx_nchw_nkhw_no_extra_padding
// CHECK-SAME: %[[filter:.*]]:  memref<1x128x64x3x3xf32>
// CHECK: %[[gemmFilter:.*]] = miopen.transform %[[filter]]
// CHECK: miopen.gridwise_gemm(%[[gemmFilter]], %{{.*}}, %{{.*}})
func.func @miopen_conv2d_kcyx_nchw_nkhw_no_extra_padding(%filter : memref<1x128x64x3x3xf32>, %input : memref<128x1x64x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    numCu = 64 : i32,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni","gi", "ci", "hi", "wi"],
    output_layout = ["no", "go",  "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<1x128x64x3x3xf32>, memref<128x1x64x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}

// CHECK-LABEL: func.func @miopen_conv2d_kcyx_nchw_nkhw_partial_padding_kernel
// CHECK-SAME: %[[filter:.*]]: memref<32x128x2x3x3xf32>
// CHECK: %[[gemmFilter:.*]] = miopen.transform %[[filter]]
// CHECK: %[[padK:.*]] = miopen.transform %[[gemmFilter]] by [#[[$PAD_GEMMK]]]
// CHECK: miopen.gridwise_gemm(%[[padK]], %{{.*}}, %{{.*}})

func.func @miopen_conv2d_kcyx_nchw_nkhw_partial_padding_kernel(%filter : memref<32x128x2x3x3xf32>, %input : memref<128x32x2x11x11xf32>, %output : memref<128x32x128x9x9xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    numCu = 64 : i32,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni","gi", "ci", "hi", "wi"],
    output_layout = ["no", "go",  "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<32x128x2x3x3xf32>, memref<128x32x2x11x11xf32>, memref<128x32x128x9x9xf32>
  return
}


