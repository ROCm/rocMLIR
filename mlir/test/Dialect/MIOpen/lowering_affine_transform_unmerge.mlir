// RUN: mlir-opt -miopen-affine-transform  %s| FileCheck %s
module  {
  func @miopen_unmerge_test(%arg0: memref<1x144x128xf32>) {
    //unmerge gemmKtotal to gemmK and gemmKpack, the dimension length is 18 and 8
    %0 = miopen.transform(%arg0) {gridwise_gemm_argument_position = 0 : i32, layout = [{dimensions = [0 : i32], names = ["gemmG"],
         source_dimensions = [0 : i32], source_names = ["gemmG"], transformation = "PassThrough"},
         {dimensions = [1 : i32, 3 : i32], names = ["gemmK", "gemmKpack"], source_dimensions = [1 : i32], 
         dimension_lengths = [18 : i32, 8 : i32], source_names = ["gemmKtotal"], transformation = "UnMerge"},{dimensions = [2 : i32], names = ["gemmM"], source_dimensions = [2 : i32], source_names = ["gemmM"], transformation = "PassThrough"}],
         intermediate_layout = ["gemmG", "gemmKtotal", "gemmM"], output_layout = ["gemmG", "gemmK", "gemmM", "gemmKpack"]} : memref<1x144x128xf32> to memref<1x18x128x8xf32>
    return
  }
}
// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 8 + d3, d2)>


