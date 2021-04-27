// This tests checks the  affinemap component:
// RUN: mlir-opt -miopen-lowering -miopen-affine-transform %s | FileCheck %s

module  {
  func @pad_parameters_00(%arg0: memref<1x128x128x3x3xf32>, %arg1: memref<256x1x128x28x28xf32>, %arg2: memref<256x1x128x26x26xf32>) attributes {kernel} {
    %0 = miopen.transform(%arg0) {extraPad = "false", gemmK_extra = 0 : i32, gemmM_extra = 0 : i32, gridwise_gemm_argument_position = 0 : i32, layout = [{dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [0 : i32], source_names = ["g"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["gemmK"], source_dimensions = [2 : i32, 3 : i32, 4 : i32], source_names = ["c", "y", "x"], transformation = "Unfold"}, {dimensions = [2 : i32], names = ["gemmM"], source_dimensions = [1 : i32], source_names = ["k"], transformation = "PassThrough"}], output_layout = ["gemmG", "gemmK", "gemmM"], source_layout = ["g", "k", "c", "y", "x"]} : memref<1x128x128x3x3xf32> to memref<1x1152x128xf32>
    %1 = miopen.transform(%arg1) {extraPad = "false", gemmK_extra = 0 : i32, gemmN_extra = 0 : i32, layout = [{dimensions = [1 : i32], names = ["gi"], source_dimensions = [1 : i32], source_names = ["gi"], transformation = "PassThrough"}, {dimensions = [0 : i32], names = ["ni"], source_dimensions = [0 : i32], source_names = ["ni"], transformation = "PassThrough"}, {dimensions = [2 : i32], names = ["ci"], source_dimensions = [2 : i32], source_names = ["ci"], transformation = "PassThrough"}, {dimensions = [3 : i32, 4 : i32], names = ["hipad", "wipad"], parameters = [1 : i32, 1 : i32], source_dimensions = [3 : i32, 4 : i32], source_names = ["hi", "wi"], transformation = "Pad"}], output_layout = ["ni", "gi", "ci", "hipad", "wipad"], source_layout = ["ni", "gi", "ci", "hi", "wi"]} : memref<256x1x128x28x28xf32> to memref<256x1x128x28x28xf32>
  return
  }

  func @pad_parameters_1101(%arg0: memref<1x128x128x3x3xf32>, %arg1: memref<256x1x128x28x28xf32>, %arg2: memref<256x1x128x26x26xf32>) attributes {kernel} {
    %0 = miopen.transform(%arg0) {extraPad = "false", gemmK_extra = 0 : i32, gemmM_extra = 0 : i32, gridwise_gemm_argument_position = 0 : i32, layout = [{dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [0 : i32], source_names = ["g"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["gemmK"], source_dimensions = [2 : i32, 3 : i32, 4 : i32], source_names = ["c", "y", "x"], transformation = "Unfold"}, {dimensions = [2 : i32], names = ["gemmM"], source_dimensions = [1 : i32], source_names = ["k"], transformation = "PassThrough"}], output_layout = ["gemmG", "gemmK", "gemmM"], source_layout = ["g", "k", "c", "y", "x"]} : memref<1x128x128x3x3xf32> to memref<1x1152x128xf32>
    %1 = miopen.transform(%arg1) {extraPad = "false", gemmK_extra = 0 : i32, gemmN_extra = 0 : i32, layout = [{dimensions = [1 : i32], names = ["gi"], source_dimensions = [1 : i32], source_names = ["gi"], transformation = "PassThrough"}, {dimensions = [0 : i32], names = ["ni"], source_dimensions = [0 : i32], source_names = ["ni"], transformation = "PassThrough"}, {dimensions = [2 : i32], names = ["ci"], source_dimensions = [2 : i32], source_names = ["ci"], transformation = "PassThrough"}, {dimensions = [3 : i32, 4 : i32], names = ["hipad", "wipad"], parameters = [1 : i32, 1 : i32, 0 : i32, 1 : i32], source_dimensions = [3 : i32, 4 : i32], source_names = ["hi", "wi"], transformation = "Pad"}], output_layout = ["ni", "gi", "ci", "hipad", "wipad"], source_layout = ["ni", "gi", "ci", "hi", "wi"]} : memref<256x1x128x28x28xf32> to memref<256x1x128x28x28xf32>
  return
  }
}

// CHECK-LABEL: #map0 =
// CHECK-NEXT: affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 - 1, d4 - 1)>
// CHECK-NEXT: affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 - 1, d4 + d4 ceildiv 28 - 1)>
