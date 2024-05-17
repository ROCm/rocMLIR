// RUN: rocmlir-gen -emit-module-fusibility-for=v2:16,16,4,16,16,1,5,1,1 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:0
// RUN: rocmlir-gen -emit-module-fusibility-for=v2:16,16,4,16,16,1,1,1,1 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
module {
  func.func @mlir_dot_add(%arg0: memref<1x2x320xf32>, %arg1: memref<1x2x1280xf32>, %arg2: memref<1x1280x320xf32>, %arg3: memref<1x2x320xf32>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x320xf32>
    rock.gemm %alloc = %arg1 * %arg2 features =  mfma|dot|atomic_add storeMethod =  set {arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} : memref<1x2x320xf32> = memref<1x2x1280xf32> * memref<1x1280x320xf32>
    %0 = rock.transform %alloc by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 2} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [2, 320] -> [1, 2, 320]> : memref<1x2x320xf32> to memref<2x320xf32>
    %1 = rock.transform %arg0 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 2} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [2, 320] -> [1, 2, 320]> : memref<1x2x320xf32> to memref<2x320xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x320xf32>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : memref<2x320xf32>, memref<2x320xf32>) outs(%alloc_0 : memref<2x320xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %3 = arith.addf %in, %in_1 : f32
      linalg.yield %3 : f32
    }
    %2 = rock.transform %alloc_0 by <affine_map<(d0, d1, d2) -> (d0 * 2 + d1, d2)> by [<Unmerge{1, 2} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 2, 320] -> [2, 320]> : memref<2x320xf32> to memref<1x2x320xf32>
    memref.copy %2, %arg3 : memref<1x2x320xf32> to memref<1x2x320xf32>
    return
  }
}
