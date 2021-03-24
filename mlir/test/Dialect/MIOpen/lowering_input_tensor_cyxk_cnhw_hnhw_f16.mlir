// This tests checks the following aspects of lowering component:
// * Input has three transformations in total
// * Input has correct output_layout across transformations

// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

func @miopen_conv2d_cyxk_cnhw_knhw(%filter : memref<8x3x3x128xf16>, %input : memref<8x128x32x32xf16>, %output : memref<128x128x30x30xf16>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["c", "y", "x", "k"],
    input_layout = ["ci", "ni", "hi", "wi"],
    output_layout = ["ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<8x3x3x128xf16>, memref<8x128x32x32xf16>, memref<128x128x30x30xf16>
  return
}
// CHECK-LABEL: func @miopen_conv2d
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK:       output_layout = ["ci", "ni", "hipad", "wipad"]
// CHECK-NEXT:  miopen.transform
// CHECK:       output_layout = ["ci", "ni", "y", "ho", "x", "wo"]
// CHECK-NEXT:  miopen.transform
// CHECK:       output_layout = ["gemmK", "gemmN"]
// CHECK-NEXT:  miopen.transform(%arg2)

func @miopen_conv2d_bwd_data_cyxk_cnhw_knhw(%filter : memref<8x3x3x128xf16>, %input : memref<8x128x32x32xf16>, %output : memref<128x128x30x30xf16>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["c", "y", "x", "k"],
    input_layout = ["ci", "ni", "hi", "wi"],
    output_layout = ["ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<8x3x3x128xf16>, memref<8x128x32x32xf16>, memref<128x128x30x30xf16>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_data
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK:       output_layout = ["ci", "ni", "hipad", "wipad"]
// CHECK-NEXT:  miopen.transform
// CHECK:       output_layout = ["ci", "ni", "y", "ho", "x", "wo"]
// CHECK-NEXT:  miopen.transform
// CHECK:       output_layout = ["gemmM", "gemmN"]
// CHECK-NEXT:  miopen.transform(%arg2)

func @miopen_conv2d_bwd_weight_cyxk_cnhw_knhw(%filter : memref<8x3x3x128xf16>, %input : memref<8x128x32x32xf16>, %output : memref<128x128x30x30xf16>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["c", "y", "x", "k"],
    input_layout = ["ci", "ni", "hi", "wi"],
    output_layout = ["ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<8x3x3x128xf16>, memref<8x128x32x32xf16>, memref<128x128x30x30xf16>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_weight
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK:       output_layout = ["ci", "ni", "hipad", "wipad"]
// CHECK-NEXT:  miopen.transform
// CHECK:       output_layout = ["ci", "ni", "y", "ho", "x", "wo"]
// CHECK-NEXT:  miopen.transform
// CHECK:       output_layout = ["gemmK", "gemmN"]
// CHECK-NEXT:  miopen.transform(%arg2)
