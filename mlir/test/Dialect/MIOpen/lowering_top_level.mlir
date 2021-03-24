// This tests checks the following aspects of lowering component:
// * Can pass arguments correctly 
// * Can pass arguments in the right sequence
// * Have the right number of transforms
// * Have one gridwise_gemm

// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

func @miopen_conv2d(%filter : memref<1x128x8x3x3xf32>, %input : memref<1x128x8x32x32xf32>, %output : memref<1x128x128x30x30xf32>) {
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
// CHECK-LABEL: func {{@miopen_conv2d.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform(%arg2)
// CHECK-NEXT:  miopen.gridwise_gemm

func @miopen_conv2d_bwd_data(%filter : memref<1x128x8x3x3xf32>, %input : memref<1x128x8x32x32xf32>, %output : memref<1x128x128x30x30xf32>) {
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
// CHECK-LABEL: func {{@miopen_conv2d_bwd_data.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_data
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform(%arg2)
// CHECK-NEXT:  miopen.gridwise_gemm

func @miopen_conv2d_bwd_weight(%filter : memref<1x128x8x3x3xf32>, %input : memref<1x128x8x32x32xf32>, %output : memref<1x128x128x30x30xf32>) {
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
// CHECK-LABEL: func {{@miopen_conv2d_bwd_weight.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_data
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform(%arg2)
// CHECK-NEXT:  miopen.gridwise_gemm
