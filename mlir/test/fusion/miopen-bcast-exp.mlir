// RUN: miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-linalg-align %s | FileCheck %s

// CHECK: #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (0, 0, 0, 0, d4)> by [#miopen.transform<PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>, #miopen.transform<Broadcast{1, 1, 1, 1} ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3] -> ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3]>] bounds = [1, 1, 30, 30, 16] -> [1, 1, 1, 1, 16]>
// CHECK-NEXT: #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d4)> by [#miopen.transform<PassThrough ["dim0"] at [4] -> ["dim0"] at [0]>, #miopen.transform<AddDim{1} ["exp0"] at [0] -> [] at []>, #miopen.transform<AddDim{1} ["exp1"] at [1] -> [] at []>, #miopen.transform<AddDim{1} ["exp2"] at [2] -> [] at []>, #miopen.transform<AddDim{1} ["exp3"] at [3] -> [] at []>] bounds = [1, 1, 1, 1, 16] -> [16]>

#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
module {
  func @test_fusion(%arg0: memref<1x1x32x32x8xf32>, %arg1: memref<1x16x3x3x8xf32>, %arg2: memref<16xf32>, %arg3: memref<1x1x30x30x16xf32>) attributes {kernel} {
    %0 = memref.alloc() : memref<1x1x30x30x16xf32>
    miopen.conv2d(%arg1, %arg0, %0) {arch = "gfx906", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["gi", "ni", "hi", "wi", "ci"], num_cu = 64 : i32, output_layout = ["go", "no", "ho", "wo", "ko"], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], strides = [1 : i32, 1 : i32], xdlopsV2 = false} : memref<1x16x3x3x8xf32>, memref<1x1x32x32x8xf32>, memref<1x1x30x30x16xf32>
    linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%0, %arg2 : memref<1x1x30x30x16xf32>, memref<16xf32>) outs(%arg3 : memref<1x1x30x30x16xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %8 = arith.addf %arg4, %arg5 : f32
      linalg.yield %8 : f32
    }
    return
  }
}
