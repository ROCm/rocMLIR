// RUN: mlir-opt -miopen-affine-transform  %s| FileCheck %s
module  {
  func @miopen_conv2d(%arg0: memref<160x192x3xf32>) {
    %0 = miopen.transform(%arg0) { layout = [{upper_layer_dimensions = [0 : i32], upper_layer_names = ["gemmG"], lower_layer_dimensions = [2 : i32], 
         lower_layer_names = ["gemmG"], transformation = "PassThrough"}, {upper_layer_dimensions = [1 : i32, 2:i32], 
         upper_layer_names = ["gemmM", "gemmN"], lower_layer_dimensions = [0 : i32, 1 : i32], lower_layer_names = ["gemmM", "gemmN"], 
         begins = [32:i32, 64:i32],ends = [160:i32, 192:i32 ], transformation = "Slice"}], 
         upper_layer_layout = ["gemmG", "gemmM", "gemmN"], lower_layer_layout = [ "gemmM", "gemmN", "gemmG"]} : memref<160x192x3xf32> to memref<3x128x128xf32>
   return
  }
}

// CHECK: #map = affine_map<(d0, d1, d2) -> (d1 + 32, d2 + 64, d0)>
