// RUN: mlir-opt -miopen-lowering -miopen-affine-transform -miopen-affix-params %s | FileCheck %s
module  {
  func @miopen_conv2d_bwd_weight_gkcyx_ngchw_ngkhw_0(%arg0: memref<1x32x32x3x3xf32>, %arg1: memref<32x1x32x7x7xf32>, %arg2: memref<32x1x32x9x9xf32>) attributes {kernel = 0 : i32} {
    miopen.conv2d_bwd_weight(%arg0, %arg1, %arg2) {arch = "gfx908", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], gemm_id = 0 : i32, input_layout = ["ni", "gi", "ci", "hi", "wi"], num_cu = 120 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [2 : i32, 2 : i32, 2 : i32, 2 : i32], strides = [1 : i32, 1 : i32], xdlopsV2 = true} : memref<1x32x32x3x3xf32>, memref<32x1x32x7x7xf32>, memref<32x1x32x9x9xf32>
    return
  }
}
// CHECK: affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5)>
// CHECK: affine_map<(d0, d1, d2, d3, d4, d5) -> (d1 * 16 + d2, d0, d3, d4 - 2, d5 - 2)>
// CHECK: affine_map<(d0, d1, d2, d3, d4, d5) -> (d1 * 16 + d2, d0, d3, d4, d5)>
// CHECK-LABEL: func @miopen_conv2d_bwd_weight_gkcyx_ngchw_ngkhw_0
// CHECK-NEXT:  {{miopen.transform.*{.*transformation = "AddDim".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*transformation = "UnMerge".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*transformation = "UnMerge".*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.transform.*{.*}.*memref.*memref}}
// CHECK-NEXT:  {{miopen.gridwise_gemm_v2.*{.*}.*memref.*memref.*memref}}