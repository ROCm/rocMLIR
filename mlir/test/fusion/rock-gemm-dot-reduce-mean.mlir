// RUN: rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-regularize -rock-gridwise-gemm-to-blockwise -rock-linalg-align -mlir-print-local-scope %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> ((d0 * 32 + d1) * 128 + d2)>
#map1 = affine_map<(d0, d1, d2) -> ((d0 * 128 + d1) * 32 + d2)>
#map2 = affine_map<(d0, d1, d2) -> ((d0 * 32 + d1) * 32 + d2)>
#map3 = affine_map<(d0, d1) -> (0, d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0 * 32 + d1, d2)>
#map6 = affine_map<(d0) -> (0, d0, 0)>
#transform_map = #rock.transform_map<#map by [<Unmerge{1, 32, 128} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 32, 128] -> [4096]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{1, 128, 32} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 128, 32] -> [4096]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{1, 32, 32} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 32, 32] -> [1024]>
#transform_map3 = #rock.transform_map<#map3 by [<Merge{1, 32} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [32, 32] -> [1, 32, 32]>
#transform_map4 = #rock.transform_map<#map5 by [<Unmerge{1, 32} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 32, 32] -> [32, 32]>
#transform_map5 = #rock.transform_map<#map6 by [<Merge{1, 32, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [32] -> [1, 32, 1]>

func.func @mlir_dot_add_reduce_mean(%arg0: memref<1024xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>, %arg3: memref<32xf32> {func.read_access, rock.prefill = 0.000000e+00 : f32}) attributes {arch = "gfx942:sramecc+:xnack-", enable_splitk_for_tuning, kernel = "mixr", num_cu = 120 : i64} {
    %cst = arith.constant 3.125000e-02 : f32
    %0 = rock.transform %arg1 by #transform_map : memref<4096xf32> to memref<1x32x128xf32>
    %1 = rock.transform %arg2 by #transform_map1 : memref<4096xf32> to memref<1x128x32xf32>
    %2 = rock.transform %arg0 by #transform_map2 : memref<1024xf32> to memref<1x32x32xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x32x32xf32>
    rock.gemm %alloc = %0 * %1 features =  mfma|dot|atomic_add storeMethod =  set {arch = "gfx942:sramecc+:xnack-", numCU = 120 : i32, perf_config = "v2:32,32,8,16,16,8,1,1,1"} : memref<1x32x32xf32> = memref<1x32x128xf32> * memref<1x128x32xf32>
    %3 = rock.transform %alloc by #transform_map3 : memref<1x32x32xf32> to memref<32x32xf32>
    %4 = rock.transform %2 by #transform_map3 : memref<1x32x32xf32> to memref<32x32xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%3, %4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_0 : memref<32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %7 = arith.addf %in, %in_2 : f32
      %8 = arith.mulf %7, %cst : f32
      linalg.yield %8 : f32
    }
    %5 = rock.transform %alloc_0 by #transform_map4 : memref<32x32xf32> to memref<1x32x32xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1xf32>
    rock.reduce  sum %5 into %alloc_1 features =  mfma|dot|atomic_add {axis = 2 : index, blockSize = 256 : i32, gridSize = 4 : i32} : memref<1x32x32xf32> into memref<1x32x1xf32>
    %6 = rock.transform %alloc_1 by #transform_map5 : memref<1x32x1xf32> to memref<32xf32>
    // CHECK: %[[OUT_VIEW0:.+]] = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d0 * 32 + d1 + d2)> by [<Unmerge{1, 32, 1} ["col0", "col1", "col2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 32, 1] -> [32]> : memref<32xf32> to memref<1x32x1xf32>
    // CHECK: %[[OUT_VIEW1:.+]] = rock.transform %[[OUT_VIEW0]] by <affine_map<(d0, d1, d2) -> (d0, d1, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [1, 32, 32] -> [1, 32, 1]> : memref<1x32x1xf32> to memref<1x32x32xf32>
    // CHECK: %[[OUT_VIEW2:.+]] = rock.transform %[[OUT_VIEW1]] by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 32} ["dim0"] at [0] -> ["exp0", "exp1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [32, 32] -> [1, 32, 32]> : memref<1x32x32xf32> to memref<32x32xf32>
    // CHECK: %[[OUT_VIEW3:.+]] = rock.transform %[[OUT_VIEW2]] by <affine_map<(d0, d1, d2) -> (d0 * 32 + d1, d2)> by [<Unmerge{1, 32} ["col0", "col1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 32, 32] -> [32, 32]> : memref<32x32xf32> to memref<1x32x32xf32>
    // rock.threadwise_write_all features =  mfma|dot|atomic_add {forceUnroll, useIndexDiffs} %{{.+}} -> [{{.*}}](%[[OUT_VIEW3]]) [{{.*}}] by  atomic_add : memref<4xf32, #gpu.address_space<private>> -> memref<1x32x32xf32>
    memref.copy %6, %arg3 : memref<32xf32> to memref<32xf32>
    return
  }
