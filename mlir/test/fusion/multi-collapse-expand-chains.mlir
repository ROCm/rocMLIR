// RUN: rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-regularize -rock-gridwise-gemm-to-blockwise -rock-linalg-align %s | FileCheck %s
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#transform_map0 = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)> by [<PassThrough ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3] -> ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3]>, <AddDim{1} ["g"] at [4] -> [] at []>] bounds = [4, 3, 3, 3, 1] -> [4, 3, 3, 3]>
#transform_map1 = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)> by [<PassThrough ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3] -> ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3]>, <AddDim{1} ["g"] at [4] -> [] at []>] bounds = [4, 4, 1, 1, 1] -> [4, 4, 1, 1]>
module {
    // CHECK-DAG: #[[MAP:.*]] = #rock.transform_map<affine_map<(d0, d1) -> (0, d1)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>] bounds = [4, 4] -> [1, 4]>
    // CHECK: rock.threadwise_read_into {{.*}}
  func.func @test(%arg0: memref<1x4x1x1xf32>, %arg1: memref<4x3x3x3xf32>, %arg2: memref<4x3x3x3xf32>, %arg3: memref<4x4x1x1xf32>) attributes {arch = "gfx908:sramecc+:xnack-", kernel = "mixr"} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x4x1x1xf32>
    %1 = rock.transform %arg1 by #transform_map0 : memref<4x3x3x3xf32> to memref<4x3x3x3x1xf32>
    %2 = rock.transform %arg2 by #transform_map0 : memref<4x3x3x3xf32> to memref<4x3x3x3x1xf32>
    %3 = rock.transform %0 by #transform_map1 : memref<4x4x1x1xf32> to memref<4x4x1x1x1xf32>
    rock.conv2d(%2, %1, %3) features =  mfma|dot|atomic_add {arch = "gfx908:sramecc+:xnack-", dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x", "g"], input_layout = ["ni", "ci", "hi", "wi", "gi"], output_layout = ["no", "ko", "ho", "wo", "go"], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<4x3x3x3x1xf32>, memref<4x3x3x3x1xf32>, memref<4x4x1x1x1xf32>
    %4 = memref.expand_shape %0 [[0], [1, 2], [3], [4]] : memref<4x4x1x1xf32> into memref<4x1x4x1x1xf32>
    %5 = memref.collapse_shape %4 [[0, 1], [2, 3, 4]] : memref<4x1x4x1x1xf32> into memref<4x4xf32>
    %6 = memref.collapse_shape %arg0 [[0, 1, 2, 3]] : memref<1x4x1x1xf32> into memref<4xf32>
    %77 = memref.alloc() : memref<4x4xf32>
    linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%5, %6 : memref<4x4xf32>, memref<4xf32>) outs(%77 : memref<4x4xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %9 = arith.addf %arg4, %arg5 : f32
      %10 = arith.minf %9, %cst_0 : f32
      %11 = arith.maxf %10, %cst : f32
      linalg.yield %11 : f32
    }
    %7 = memref.expand_shape %77 [[0, 1], [2, 3, 4]] : memref<4x4xf32> into memref<4x1x4x1x1xf32>
    %8 = memref.collapse_shape %7 [[0], [1, 2], [3], [4]] : memref<4x1x4x1x1xf32> into memref<4x4x1x1xf32>
    memref.copy %8, %arg3 : memref<4x4x1x1xf32> to memref<4x4x1x1xf32>
    return
  }
}
