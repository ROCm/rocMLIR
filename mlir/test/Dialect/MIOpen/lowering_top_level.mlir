// This tests checks the following aspects of lowering component:
// * Can pass arguments correctly 
// * Can pass arguments in the right sequence
// * Have the right number of transforms
// * Have one gridwise_gemm

// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

func @miopen_conv2d(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["k", "c", "y", "x"],
    input_layout = ["ni", "ci", "hi", "wi"],
    output_layout = ["no", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
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

func @miopen_conv2d_bwd_data(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["k", "c", "y", "x"],
    input_layout = ["ni", "ci", "hi", "wi"],
    output_layout = ["no", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
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

func @miopen_conv2d_bwd_weight(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["k", "c", "y", "x"],
    input_layout = ["ni", "ci", "hi", "wi"],
    output_layout = ["no", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
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
