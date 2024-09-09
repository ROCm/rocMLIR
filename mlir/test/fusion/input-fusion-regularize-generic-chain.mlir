// RUN: rocmlir-opt --rock-regularize --mlir-print-local-scope %s | FileCheck %s

#general_gemm_params = #rock.general_gemm_params<blockSize = 128, kPerBlock = 16, mPerBlock = 32, nPerBlock = 32, kPerThread = 1, mPerThread = 2, nPerThread = 2, kpack = 1, splitKFactor = 1>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#transform_map = #rock.transform_map<#map1 by [<PassThrough ["gemmG", "gemmK", "gemmM"] at [0, 1, 2] -> ["gemmG", "gemmK", "gemmM"] at [0, 2, 1]>] bounds = [1, 32, 16] -> [1, 16, 32]>

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx90a"} {
  // CHECK-LABEL: @rock_gemm
  func.func @rock_gemm(%arg0: memref<1x32x16xf16>, %arg1: memref<1x16x32xf32>, %arg2: memref<1x32x32xf32>) attributes {block_size = 128 : i32, enable_splitk_for_tuning, grid_size = 1 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a"} {
    // CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<1x32x16xf32>
    %alloc = memref.alloc() : memref<1x32x16xf32>
    // CHECK-NEXT: linalg.generic {{.*}} outs(%[[ALLOC]]
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x32x16xf16>) outs(%alloc : memref<1x32x16xf32>) {
    ^bb0(%in: f16, %out: f32):
      %1 = arith.extf %in : f16 to f32
      linalg.yield %1 : f32
    }
    // CHECK: %[[ALLOC_0:.+]] = memref.alloc() : memref<1x32x16xf32>
    %alloc_0 = memref.alloc() : memref<1x32x16xf32>
    // CHECK-NEXT: linalg.generic {{.*}} ins(%[[ALLOC]] : memref<1x32x16xf32>) outs(%[[ALLOC_0]]
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<1x32x16xf32>) outs(%alloc_0 : memref<1x32x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst = arith.constant 2.000000e+00 : f32
      %1 = arith.addf %in, %cst : f32
      linalg.yield %1 : f32
    }
    // CHECK: %[[IN:.+]] = rock.transform %[[ALLOC_0]]
    // CHECK-SAME: [<PassThrough ["gemmG", "gemmK", "gemmM"] at [0, 1, 2] -> ["gemmG", "gemmK", "gemmM"] at [0, 2, 1]>]
    %0 = rock.transform %alloc_0 by #transform_map : memref<1x32x16xf32> to memref<1x16x32xf32>
    // CHECK-NEXT: rock.gridwise_gemm %{{.*}} = %[[IN]] * %{{.*}}
    rock.gridwise_gemm %arg2 = %0 * %arg1 storeMethod(set) features =  dot|atomic_add {gridSize = 1 : i32, numCU = 104 : i32, params = #general_gemm_params} : memref<1x32x32xf32> = memref<1x16x32xf32> * memref<1x16x32xf32>
    return
  }


  // CHECK-LABEL: @rock_gemm_tr
  func.func @rock_gemm_tr(%arg0: memref<1x32x16xf16>, %arg1: memref<1x16x32xf32>, %arg2: memref<1x32x32xf32>) attributes {block_size = 128 : i32, enable_splitk_for_tuning, grid_size = 1 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a"} {
    // CHECK: %[[alloc:.+]] = memref.alloc() : memref<1x32x16xf32>
    // CHECK-NEXT: %[[trAlloc:.*]] = rock.transform %[[alloc]] by
    // CHECK-SAME: [<PassThrough ["gemmG", "gemmK", "gemmM"] at [0, 2, 1] -> ["gemmG", "gemmK", "gemmM"] at [0, 1, 2]>]
    %alloc = memref.alloc() : memref<1x16x32xf32>
    %0 = rock.transform %alloc by #transform_map : memref<1x16x32xf32> to memref<1x32x16xf32>
    // CHECK-NEXT: linalg.generic {{.*}} outs(%[[alloc]]
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x32x16xf16>) outs(%0 : memref<1x32x16xf32>) {
    ^bb0(%in: f16, %out: f32):
      %3 = arith.extf %in : f16 to f32
      linalg.yield %3 : f32
    }
    // CHECK: %[[alloc_0:.+]] = memref.alloc() : memref<1x32x16xf32>
    // CHECK-NEXT: %[[gemmIn:.+]] = rock.transform %[[alloc_0]]
    // CHECK-SAME: [<PassThrough ["gemmG", "gemmK", "gemmM"] at [0, 2, 1] -> ["gemmG", "gemmK", "gemmM"] at [0, 1, 2]>]
    // CHECK-NEXT: %[[reTrAlloc:.+]] = rock.transform %[[trAlloc]]
    // CHECK-SAME: [<PassThrough ["gemmG", "gemmK", "gemmM"] at [0, 1, 2] -> ["gemmG", "gemmK", "gemmM"] at [0, 2, 1]>]
    %alloc_0 = memref.alloc() : memref<1x16x32xf32>
    %1 = rock.transform %alloc_0 by #transform_map : memref<1x16x32xf32> to memref<1x32x16xf32>
    %2 = rock.transform %alloc by #transform_map : memref<1x16x32xf32> to memref<1x32x16xf32>
    // CHECK-NEXT: linalg.generic
    // CHECK-SAME: ins(%[[reTrAlloc]] : memref<1x32x16xf32>) outs(%[[alloc_0]]
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : memref<1x32x16xf32>) outs(%1 : memref<1x32x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst = arith.constant 2.000000e+00 : f32
      %3 = arith.addf %in, %cst : f32
      linalg.yield %3 : f32
    }
    // CHECK: rock.gridwise_gemm %{{.+}} = %[[gemmIn]] * %{{.+}}
    rock.gridwise_gemm %arg2 = %alloc_0 * %arg1 storeMethod(set) features =  dot|atomic_add {gridSize = 1 : i32, numCU = 104 : i32, params = #general_gemm_params} : memref<1x32x32xf32> = memref<1x16x32xf32> * memref<1x16x32xf32>
    return
  }
}
