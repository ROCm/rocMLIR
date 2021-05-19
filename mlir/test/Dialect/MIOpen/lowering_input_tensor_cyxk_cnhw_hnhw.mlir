// This tests checks the following aspects of lowering component:
// * Input has three transformations in total
// * Input has correct output_layout across transformations

// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

func @miopen_conv2d_cyxk_cnhw_knhw(%filter : memref<1x8x3x3x128xf32>, %input : memref<1x8x128x32x32xf32>, %output : memref<1x128x128x30x30xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "c", "y", "x", "k"],
    input_layout = ["gi", "ci", "ni", "hi", "wi"],
    output_layout = ["go", "ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0, 0]
  } : memref<1x8x3x3x128xf32>, memref<1x8x128x32x32xf32>, memref<1x128x128x30x30xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK:       output_layout = ["gi", "ci", "ni", "hipad", "wipad"]
// CHECK-NEXT:  miopen.transform
// CHECK:       output_layout = ["gi", "ci", "ni", "y", "ho", "x", "wo"]
// CHECK-NEXT:  miopen.transform
// CHECK:       output_layout = ["gemmG", "gemmK", "gemmN"]
// CHECK-NEXT:  miopen.transform(%arg2)

func @miopen_conv2d_bwd_data_cyxk_cnhw_knhw(%filter : memref<1x8x3x3x128xf32>, %input : memref<1x8x128x32x32xf32>, %output : memref<1x128x128x30x30xf32>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "c", "y", "x", "k"],
    input_layout = ["gi", "ci", "ni", "hi", "wi"],
    output_layout = ["go", "ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0, 0]
  } : memref<1x8x3x3x128xf32>, memref<1x8x128x32x32xf32>, memref<1x128x128x30x30xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_data
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK:       miopen.transform(%arg1)
// CHECK:       output_layout = ["gi", "ni", "ci", "hipad", "wipad"]
// CHECK-NEXT:  miopen.transform
// CHECK:       output_layout = ["gi", "ni", "ci", "ytilda", "htilda", "xtilda", "wtilda"]
// CHECK-NEXT:  miopen.transform
// CHECK:       output_layout = ["gemmG", "gemmM", "gemmN"]
// CHECK-NEXT:  miopen.transform(%arg2)

func @miopen_conv2d_bwd_weight_cyxk_cnhw_knhw(%filter : memref<1x8x3x3x128xf32>, %input : memref<1x8x128x32x32xf32>, %output : memref<1x128x128x30x30xf32>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "c", "y", "x", "k"],
    input_layout = ["gi", "ci", "ni", "hi", "wi"],
    output_layout = ["go", "ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0, 0]
  } : memref<1x8x3x3x128xf32>, memref<1x8x128x32x32xf32>, memref<1x128x128x30x30xf32>
  return
}
// CHECK-LABEL: func @miopen_conv2d_bwd_weight
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK:       output_layout = ["gi", "ci", "ni", "hipad", "wipad"]
// CHECK-NEXT:  miopen.transform
// CHECK:       output_layout = ["gi", "ci", "ni", "y", "ho", "x", "wo"]
// CHECK-NEXT:  miopen.transform
// CHECK:       output_layout = ["gemmG", "gemmK", "gemmN"]
// CHECK-NEXT:  miopen.transform(%arg2)
