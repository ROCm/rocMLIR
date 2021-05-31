// This tests checks the  affinemap component:
// RUN: mlir-opt -miopen-lowering -miopen-affine-transform %s | FileCheck %s

 module  {
   func @pad_parameters_1101(%arg0: memref<256x1x128x28x28xf32>) attributes {kernel} {
    %1 = miopen.transform(%arg0) {extraPad = "false", gemmK_extra = 0 : i32, gemmN_extra = 0 : i32, layout = [{upper_layer_dimensions = [1 : i32], upper_layer_names = ["gi"], lower_layer_dimensions = [1 : i32], lower_layer_names = ["gi"], transformation = "PassThrough"}, {upper_layer_dimensions = [0 : i32], upper_layer_names = ["ni"], lower_layer_dimensions = [0 : i32], lower_layer_names = ["ni"], transformation = "PassThrough"}, {upper_layer_dimensions = [2 : i32], upper_layer_names = ["ci"], lower_layer_dimensions = [2 : i32], lower_layer_names = ["ci"], transformation = "PassThrough"}, {upper_layer_dimensions = [3 : i32, 4 : i32], upper_layer_names = ["hipad", "wipad"], parameters = [1 : i32, 1 : i32, 0 : i32, 1 : i32], lower_layer_dimensions = [3 : i32, 4 : i32], lower_layer_names = ["hi", "wi"], transformation = "Pad"}], upper_layer_layout = ["ni", "gi", "ci", "hipad", "wipad"], lower_layer_layout = ["ni", "gi", "ci", "hi", "wi"]} : memref<256x1x128x28x28xf32> to memref<256x1x128x28x28xf32>
    return
   }

// CHECK: affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 - 1, (d4 + 2) ceildiv 29 + d4 - 1)>

  func @pad_parameters_00(%arg0: memref<256x1x67x28x28xf32>) attributes {kernel} {
    %1 = miopen.transform(%arg0) {extraPad = "true", gemmK_extra = 61 : i32, gemmN_extra = 0 : i32, layout = [{upper_layer_dimensions = [1 : i32], upper_layer_names = ["gi"], lower_layer_dimensions = [1 : i32], lower_layer_names = ["gi"], transformation = "PassThrough"}, {upper_layer_dimensions = [0 : i32], upper_layer_names = ["ni"], lower_layer_dimensions = [0 : i32], lower_layer_names = ["ni"], transformation = "PassThrough"}, {upper_layer_dimensions = [2 : i32], upper_layer_names = ["ci"], parameters = [0 : i32, 61 : i32],lower_layer_dimensions = [2 : i32], lower_layer_names = ["ci"], transformation = "Pad"}, {upper_layer_dimensions = [3 : i32, 4 : i32], upper_layer_names = ["hipad", "wipad"], parameters = [0 : i32, 0 : i32, 0 : i32, 0 : i32], lower_layer_dimensions = [3 : i32, 4 : i32], lower_layer_names = ["hi", "wi"], transformation = "Pad"}], upper_layer_layout = ["ni", "gi", "ci", "hipad", "wipad"], lower_layer_layout = ["ni", "gi", "ci", "hi", "wi"]} : memref<256x1x67x28x28xf32> to memref<256x1x128x28x28xf32>
   return
   }
// CHECK: affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, (d2 + 2) ceildiv 68 + d2 - 1, d3, d4)>
 }
