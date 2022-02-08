// This tests checks the following aspects of lowering component:
// * The correct padding attributes are generated and attached to the GEMM

// RUN: miopen-opt -miopen-affix-params -miopen-lowering %s | FileCheck %s
// CHECK-DAG: #[[$PAD_K:gemm_padding[0-9]+]] = #miopen.padding_info<extraM = 0, extraK = 14, extraN = 0, bwdPaddingInfo = "NA">
// CHECK-DAG: #[[$PAD_NONE:gemm_padding[0-9]+]] = #miopen.padding_info<extraM = 0, extraK = 0, extraN = 0, bwdPaddingInfo = "NA">
func @miopen_conv2d_kcyx_nchw_nkhw_padding_kernel(%filter : memref<32x128x2x3x3xf32>, %input : memref<64x32x2x11x11xf32>, %output : memref<64x32x128x9x9xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni","gi", "ci", "hi", "wi"],
    output_layout = ["no", "go",  "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<32x128x2x3x3xf32>, memref<64x32x2x11x11xf32>, memref<64x32x128x9x9xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d
// CHECK:  miopen.gridwise_gemm{{.*}}paddingInfo = #[[$PAD_K]]

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
// CHECK: miopen.gridwise_gemm{{.*}}paddingInfo = #[[$PAD_NONE]]

func @miopen_conv2d_kcyx_nchw_nkhw_partial_padding_kernel(%filter : memref<32x128x2x3x3xf32>, %input : memref<128x32x2x11x11xf32>, %output : memref<128x32x128x9x9xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni","gi", "ci", "hi", "wi"],
    output_layout = ["no", "go",  "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<32x128x2x3x3xf32>, memref<128x32x2x11x11xf32>, memref<128x32x128x9x9xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d
// CHECK:  miopen.gridwise_gemm{{.*}}paddingInfo = #[[$PAD_K]]


