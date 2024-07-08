// RUN: rocmlir-driver --rock-regularize %s | FileCheck %s

#general_gemm_params = #rock.general_gemm_params<blockSize = 128, kPerBlock = 16, mPerBlock = 32, nPerBlock = 32, kPerThread = 1, mPerThread = 2, nPerThread = 2, kpack = 1, splitKFactor = 1>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP:.*]] = #rock.transform_map<#map{{.*}} by [<PassThrough ["gemmG", "gemmK", "gemmM"] at [0, 2, 1] -> ["gemmG", "gemmK", "gemmM"] at [0, 1, 2]>] bounds = [1, 16, 32] -> [1, 32, 16]>
#transform_map = #rock.transform_map<#map by [<PassThrough ["gemmG", "gemmK", "gemmM"] at [0, 1, 2] -> ["gemmG", "gemmK", "gemmM"] at [0, 2, 1]>] bounds = [1, 32, 16] -> [1, 16, 32]>
// CHECK-DAG: #[[MAP1:.*]] = #rock.transform_map<#map{{.*}} by [<PassThrough ["gemmG", "gemmM", "gemmK"] at [0, 1, 2] -> ["gemmG", "gemmM", "gemmK"] at [0, 1, 2]>] bounds = [1, 16, 32] -> [1, 16, 32]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["gemmG", "gemmM", "gemmK"] at [0, 1, 2] -> ["gemmG", "gemmM", "gemmK"] at [0, 1, 2]>] bounds = [1, 16, 32] -> [1, 16, 32]>
// CHECK-DAG: #[[MAP2:.*]] = #rock.transform_map<#map{{.*}} by [<PassThrough ["gemmG", "gemmM", "gemmK"] at [0, 1, 2] -> ["gemmG", "gemmM", "gemmK"] at [0, 1, 2]>] bounds = [1, 32, 32] -> [1, 32, 32]>
#transform_map2 = #rock.transform_map<#map1 by [<PassThrough ["gemmG", "gemmM", "gemmK"] at [0, 1, 2] -> ["gemmG", "gemmM", "gemmK"] at [0, 1, 2]>] bounds = [1, 32, 32] -> [1, 32, 32]>
module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx90a"} {
  func.func @rock_gemm(%arg0: memref<1x32x16xf16>, %arg1: memref<1x16x32xf32>, %arg2: memref<1x32x32xf32>) attributes {block_size = 128 : i32, enable_splitk_for_tuning, grid_size = 1 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a"} {
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x32x16xf32>
    // CHECK-NEXT: %[[TRAN:.*]] = rock.transform %[[ALLOC]] by #[[MAP]]
    %alloc = memref.alloc() : memref<1x16x32xf32>
    %0 = rock.transform %alloc by #transform_map : memref<1x16x32xf32> to memref<1x32x16xf32>
    // CHECK-NEXT: linalg.generic
    // CHECK-SAME: outs(%[[ALLOC]]
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x32x16xf16>) outs(%0 : memref<1x32x16xf32>) {
    ^bb0(%in: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      linalg.yield %2 : f32
    }
    // CHECK: %[[IN:.*]]  = rock.transform %[[TRAN]] by #[[MAP1]]
    %1 = rock.transform %alloc by #transform_map1 : memref<1x16x32xf32> to memref<1x16x32xf32>
    // CHECK-NEXT: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x32x32xf32>
    %gemmOut = memref.alloc() : memref<1x32x32xf32>
    // CHECK-NEXT: rock.gridwise_gemm %[[ALLOC_0]] = %[[IN]] * %{{.*}}
    rock.gridwise_gemm %gemmOut = %1 * %arg1 features =  dot|atomic_add {gridSize = 1 : i32, numCU = 104 : i32, params = #general_gemm_params} : memref<1x32x32xf32> = memref<1x16x32xf32> * memref<1x16x32xf32>
    // CHECK-NEXT: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x32x32xf32>
    // CHECK-NEXT: %[[OUTS:.*]] = rock.transform %alloc_1 by #[[MAP2]]
    %alloc_1 = memref.alloc() : memref<1x32x32xf32>
    // CHECK-NEXT: linalg.generic
    // CHECK-SAME: outs(%[[OUTS]]
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%gemmOut : memref<1x32x32xf32>) outs(%alloc_1 : memref<1x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %c0 = arith.constant 2.0 : f32
      %r = arith.addf %in, %c0: f32
      linalg.yield %r : f32
    }
    %cp = rock.transform %alloc_1 by #transform_map2 : memref<1x32x32xf32> to memref<1x32x32xf32>
    // CHECK: memref.copy %[[ALLOC_1]]
    memref.copy %cp, %arg2 : memref<1x32x32xf32> to memref<1x32x32xf32>
    return
  }
}
