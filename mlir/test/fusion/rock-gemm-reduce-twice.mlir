// RUN: rocmlir-driver -c -arch %arch %s
// XFAIL: *
// COM: nested reductions like reduce(reduce(x)) are not allowed

#map = affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 64 + d2)>
#map1 = affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 4096 + d2)>
#map2 = affine_map<(d0, d1, d2) -> ((d0 * 4096 + d1) * 64 + d2)>
#map3 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map4 = affine_map<(d0, d1) -> (d0 floordiv 64, d0 mod 64, d1)>
#map5 = affine_map<(d0, d1) -> (0, d0, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map7 = affine_map<(d0) -> (d0, 0, 0)>
#transform_map = #rock.transform_map<#map by [<Unmerge{64, 64, 64} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [64, 64, 64] -> [262144]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{64, 64, 4096} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [64, 64, 4096] -> [16777216]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{1, 4096, 64} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 4096, 64] -> [262144]>
#transform_map3 = #rock.transform_map<#map3 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [64, 4096, 64] -> [1, 4096, 64]>
#transform_map4 = #rock.transform_map<#map4 by [<Merge{64, 64} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [4096, 4096] -> [64, 64, 4096]>
#transform_map5 = #rock.transform_map<#map5 by [<ConstDim{0, 64} [] at [] -> ["g"] at [0]>, <PassThrough ["d0", "d1"] at [0, 1] -> ["d0", "d1"] at [1, 2]>] bounds = [4096, 64] -> [64, 4096, 64]>
#transform_map6 = #rock.transform_map<#map4 by [<Merge{64, 64} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [4096, 64] -> [64, 64, 64]>
#transform_map7 = #rock.transform_map<#map7 by [<Merge{64, 1, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [64] -> [64, 1, 1]>
module {
  func.func @matmul_broadcast_op(%arg0: memref<262144xf32>, %arg1: memref<16777216xf32>, %arg2: memref<262144xf32>, %arg3: memref<64xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32}) attributes {arch = "gfx942:sramecc+:xnack-", enable_splitk_for_tuning, kernel = "mixr"} {
    %0 = rock.transform %arg0 by #transform_map : memref<262144xf32> to memref<64x64x64xf32>
    %1 = rock.transform %arg1 by #transform_map1 : memref<16777216xf32> to memref<64x64x4096xf32>
    %2 = rock.transform %arg2 by #transform_map2 : memref<262144xf32> to memref<1x4096x64xf32>
    %3 = rock.transform %2 by #transform_map3 : memref<1x4096x64xf32> to memref<64x4096x64xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64x64xf32>
    %4 = rock.transform %1 by #transform_map4 : memref<64x64x4096xf32> to memref<4096x4096xf32>
    %5 = rock.transform %3 by #transform_map5 : memref<64x4096x64xf32> to memref<4096x64xf32>
    %6 = rock.transform %alloc by #transform_map6 : memref<64x64x64xf32> to memref<4096x64xf32>
    rock.gemm %6 = %4 * %5 features =  mfma|dot|atomic_add storeMethod =  set {arch = "gfx942:sramecc+:xnack-", perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : memref<4096x64xf32> = memref<4096x4096xf32> * memref<4096x64xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64x64xf32>
    linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc, %0 : memref<64x64x64xf32>, memref<64x64x64xf32>) outs(%alloc_0 : memref<64x64x64xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %8 = arith.addf %in, %in_3 : f32
      linalg.yield %8 : f32
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<64x64x1xf32>
    rock.reduce  sum %alloc_0 into %alloc_1 features =  mfma|dot|atomic_add {axis = 2 : index, blockSize = 256 : i32, gridSize = 1024 : i32} : memref<64x64x64xf32> into memref<64x64x1xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<64x1x1xf32>
    rock.reduce  sum %alloc_1 into %alloc_2 features =  mfma|dot|atomic_add {axis = 1 : index, blockSize = 256 : i32, gridSize = 16 : i32} : memref<64x64x1xf32> into memref<64x1x1xf32>
    %7 = rock.transform %alloc_2 by #transform_map7 : memref<64x1x1xf32> to memref<64xf32>
    memref.copy %7, %arg3 : memref<64xf32> to memref<64xf32>
    return
  }
}
