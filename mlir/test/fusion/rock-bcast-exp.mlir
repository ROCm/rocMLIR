// RUN: rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-regularize -rock-gridwise-gemm-to-blockwise -rock-linalg-align %s | FileCheck %s

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
// CHECK: #rock.transform_map<#[[MAP]] by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <AddDim{1} ["exp1"] at [1] -> [] at []>, <AddDim{30} ["exp2"] at [2] -> [] at []>, <AddDim{30} ["exp3"] at [3] -> [] at []>, <PassThrough ["dim0"] at [4] -> ["dim0"] at [0]>] bounds = [1, 1, 30, 30, 16] -> [16]>

#transform_map1 = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d4)> by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <AddDim{1} ["exp1"] at [1] -> [] at []>, <AddDim{30} ["exp2"] at [2] -> [] at []>, <AddDim{30} ["exp3"] at [3] -> [] at []>, <PassThrough ["dim0"] at [4] -> ["dim0"] at [0]>] bounds = [1, 1, 30, 30, 16] -> [16]>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
module {
  func.func @test_fusion(%arg0: memref<1x1x32x32x8xf32>, %arg1: memref<1x16x3x3x8xf32>, %arg2: memref<16xf32>, %arg3: memref<1x1x30x30x16xf32>) attributes {kernel, arch = ""} {
    %0 = memref.alloc() : memref<1x1x30x30x16xf32>
    rock.conv(%arg1, %arg0, %0) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["gi", "ni", "0i", "1i", "ci"], output_layout = ["go", "no", "0o", "1o", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x16x3x3x8xf32>, memref<1x1x32x32x8xf32>, memref<1x1x30x30x16xf32>
    %1 = rock.transform %arg2 by #transform_map1 : memref<16xf32> to memref<1x1x30x30x16xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%0, %1 : memref<1x1x30x30x16xf32>, memref<1x1x30x30x16xf32>) outs(%arg3 : memref<1x1x30x30x16xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %8 = arith.addf %arg4, %arg5 : f32
      linalg.yield %8 : f32
    }
    return
  }
}
