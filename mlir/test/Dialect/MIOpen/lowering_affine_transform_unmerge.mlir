// RUN: mlir-opt -miopen-affine-transform  %s| FileCheck %s
module  {
  func @miopen_conv2d_1(%arg0: memref<1x128x16x1x1xf32>) {
    %0 = miopen.transform(%arg0) { layout = [{dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [0 : i32], source_names = ["g"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["gemmKtotal"], source_dimensions = [2 : i32, 3 : i32, 4 : i32], source_names = ["c", "y", "x"], transformation = "Unfold"}, {dimensions = [2 : i32], names = ["gemmM"], source_dimensions = [1 : i32], source_names = ["k"], transformation = "PassThrough"}], output_layout = ["gemmG", "gemmKtotal", "gemmM"], source_layout = ["g", "k", "c", "y", "x"]} : memref<1x128x16x1x1xf32> to memref<1x16x128xf32>
    %1 = miopen.transform(%0) {gridwise_gemm_argument_position = 0 : i32, layout = [{dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [0 : i32], source_names = ["gemmG"], transformation = "PassThrough"}, {dimensions = [1 : i32, 3 : i32], names = ["gemmK", "gemmKpack"], source_dimensions = [1 : i32], source_names = ["gemmKtotal"], transformation = "UnMerge"}, {dimensions = [2 : i32], names = ["gemmM"], source_dimensions = [2 : i32], source_names = ["gemmM"], transformation = "PassThrough"}], intermediate_layout = ["gemmG", "gemmKtotal", "gemmM"], output_layout = ["gemmG", "gemmk", "gemmM", "gemmKpack"]} : memref<1x16x128xf32> to memref<1x8x128x2xf32>
    return
  }
  func @miopen_conv2d_2(%arg0: memref<1x128x16x3x3xf32>) {
    %0 = miopen.transform(%arg0) { layout = [{dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [0 : i32], source_names = ["g"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["gemmKtotal"], source_dimensions = [2 : i32, 3 : i32, 4 : i32], source_names = ["c", "y", "x"], transformation = "Unfold"}, {dimensions = [2 : i32], names = ["gemmM"], source_dimensions = [1 : i32], source_names = ["k"], transformation = "PassThrough"}], output_layout = ["gemmG", "gemmKtotal", "gemmM"], source_layout = ["g", "k", "c", "y", "x"]} : memref<1x128x16x3x3xf32> to memref<1x144x128xf32>
    %1 = miopen.transform(%0) {gridwise_gemm_argument_position = 0 : i32, layout = [{dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [0 : i32], source_names = ["gemmG"], transformation = "PassThrough"}, {dimensions = [1 : i32, 3 : i32], names = ["gemmK", "gemmKpack"], source_dimensions = [1 : i32], source_names = ["gemmKtotal"], transformation = "UnMerge"}, {dimensions = [2 : i32], names = ["gemmM"], source_dimensions = [2 : i32], source_names = ["gemmM"], transformation = "PassThrough"}], intermediate_layout = ["gemmG", "gemmKtotal", "gemmM"], output_layout = ["gemmG", "gemmk", "gemmM", "gemmKpack"]} : memref<1x144x128xf32> to memref<1x18x128x8xf32>
    return
  }
}
// CHECK: #map0 = affine_map<(d0, d1, d2) -> (d0, d2, d1, 0, 0)>
// CHECK_NEXT: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1 * 2 + d3, 0, 0)>
// CHECK_NEXT: #map2 = affine_map<(d0, d1, d2) -> (d0, d2, d1 floordiv 9, (d1 mod 9) floordiv 3, (d1 mod 9) mod 3)>
// CHECK_NEXT: #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d2, (d1 * 8 + d3) floordiv 9, ((d1 * 8 + d3) mod 9) floordiv 3, ((d1 * 8 + d3) mod 9) mod 3)>