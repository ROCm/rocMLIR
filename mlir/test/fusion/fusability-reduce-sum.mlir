// RUN: rocmlir-gen -emit-module-fusibility-for=v3:16,16,4,16,16,1,5,1,2,1,1 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:1
// RUN: rocmlir-gen -emit-module-fusibility-for=v3:16,16,4,16,16,1,1,1,2,1,1 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
module {
  func.func @mlir_dot_add(%arg0: memref<1x2x320xf32>, %arg1: memref<1x2x1280xf32>, %arg2: memref<1x1280x320xf32>, %arg3: memref<1x2x1xf32>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x320xf32>
    rock.gemm %alloc = %arg1 * %arg2 storeMethod =  set : memref<1x2x320xf32> = memref<1x2x1280xf32> * memref<1x1280x320xf32>
    %0 = rock.transform %alloc by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 2} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [2, 320] -> [1, 2, 320]> : memref<1x2x320xf32> to memref<2x320xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x1xf32>
    rock.reduce  sum %0 into %alloc_1 {axis = 1 : index, blockSize = 256 : i32, gridSize = 2 : i32} : memref<2x320xf32> into memref<2x1xf32>
    %2 = rock.transform %alloc_1 by <affine_map<(d0, d1, d2) -> (d0 * 2 + d1, d2)> by [<Unmerge{1, 2} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 2, 1] -> [2, 1]> : memref<2x1xf32> to memref<1x2x1xf32>
    memref.copy %2, %arg3 : memref<1x2x1xf32> to memref<1x2x1xf32>
    return
  }
}
