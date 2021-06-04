// RUN: mlir-opt -miopen-affine-transform  %s| FileCheck %s
module  {
  func @miopen_unmerge_test(%arg0: memref<1x144x128xf32>) {
    //unmerge gemmKtotal to gemmK and gemmKpack, the dimension length is 18 and 8
    %0 = miopen.transform(%arg0) {gridwise_gemm_argument_position = 0 : i32, layout = [{upper_layer_dimensions = [0 : i32], upper_layer_names = ["gemmG"],
         lower_layer_dimensions = [0 : i32], lower_layer_names = ["gemmG"], transformation = "PassThrough"},
         {upper_layer_dimensions = [1 : i32, 3 : i32], upper_layer_names = ["gemmK", "gemmKpack"], lower_layer_dimensions = [1 : i32], 
         dimension_lengths = [18 : i32, 8 : i32], lower_layer_names = ["gemmKtotal"], transformation = "UnMerge"},{upper_layer_dimensions = [2 : i32], upper_layer_names = ["gemmM"], lower_layer_dimensions = [2 : i32], lower_layer_names = ["gemmM"], transformation = "PassThrough"}],
         lower_layer_layout = ["gemmG", "gemmKtotal", "gemmM"], upper_layer_layout = ["gemmG", "gemmK", "gemmM", "gemmKpack"]} : memref<1x144x128xf32> to memref<1x18x128x8xf32>
    return
  }
}
// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 8 + d3, d2)>


