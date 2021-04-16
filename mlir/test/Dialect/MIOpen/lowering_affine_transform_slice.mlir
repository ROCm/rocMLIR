// RUN: mlir-opt -miopen-affine-transform  %s| FileCheck %s
module  {
  func @miopen_conv2d(%arg0: memref<160x192x3xf32>) {
    %0 = miopen.transform(%arg0) { layout = [{dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [2 : i32], source_names = ["gemmG"], transformation = "PassThrough"}, {dimensions = [1 : i32, 2:i32], names = ["gemmM", "gemmN"], source_dimensions = [0 : i32, 1 : i32], source_names = ["gemmM", "gemmN"], begins = [32:i32, 64:i32],ends = [160:i32, 192:i32 ], transformation = "Slice"}], output_layout = ["gemmG", "gemmM", "gemmN"], source_layout = [ "gemmM", "gemmN", "gemmG"]} : memref<160x192x3xf32> to memref<3x128x128xf32>
   return
  }
}

// CHECK: #map = affine_map<(d0, d1, d2) -> (d1 + 32, d2 + 64, d0)>