// This tests checks the following aspects of lowering component:
// * transform has the right number of memref
// * gridwise_gemm has the right number of memref

// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

func @miopen_conv2d_kcyx_nchw_nkhw(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
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
// CHECK-LABEL: func @miopen_conv2d
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.gridwise_gemm.*{.*}.*memref.*memref.*memref}}

func @miopen_conv2d_bwd_data_kcyx_nchw_nkhw(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
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
// CHECK-LABEL: func @miopen_conv2d_bwd_data
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.gridwise_gemm.*{.*}.*memref.*memref.*memref}}

func @miopen_conv2d_bwd_weight_kcyx_nchw_nkhw(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
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
// CHECK-LABEL: func @miopen_conv2d_bwd_weight
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.gridwise_gemm.*{.*}.*memref.*memref.*memref}}
