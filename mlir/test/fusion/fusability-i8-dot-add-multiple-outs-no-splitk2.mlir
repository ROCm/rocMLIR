// RUN: rocmlir-gen -emit-module-fusibility-for=v3:16,16,4,16,16,1,5,1,2,1,1 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:0
// RUN: rocmlir-gen -emit-module-fusibility-for=v3:16,16,4,16,16,1,1,1,2,1,1 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
module {
  func.func @mlir_dot_add(%arg0: memref<1x2x320xf16>, %arg1: memref<1x2x1280xi8>, %arg2: memref<1x1280x320xi8>, %arg3: memref<1x2x320xf16>, %arg4: memref<1x2x1xf16>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x320xi8>
    rock.gemm %alloc = %arg1 * %arg2 storeMethod =  set {arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} : memref<1x2x320xi8> = memref<1x2x1280xi8> * memref<1x1280x320xi8>
    %0 = rock.transform %alloc by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 2} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [2, 320] -> [1, 2, 320]> : memref<1x2x320xi8> to memref<2x320xi8>
    %1 = rock.transform %arg0 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 2} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [2, 320] -> [1, 2, 320]> : memref<1x2x320xf16> to memref<2x320xf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x320xf16>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : memref<2x320xi8>, memref<2x320xf16>) outs(%alloc_0 : memref<2x320xf16>) {
    ^bb0(%in: i8, %in_1: f16, %out: f16):
      %3 = arith.sitofp %in : i8 to f16
      %4 = arith.addf %3, %in_1 : f16
      linalg.yield %4 : f16
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x320xf16>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : memref<2x320xi8>, memref<2x320xf16>) outs(%alloc_1 : memref<2x320xf16>) {
    ^bb0(%in: i8, %in_1: f16, %out: f16):
      %5 = arith.fptoui %in_1 : f16 to i8
      %6 = arith.addi %5, %in : i8
      %7 = arith.sitofp %6 : i8 to f16
      linalg.yield %7 : f16
    }
    %8 = rock.transform %alloc_1 by <affine_map<(d0, d1, d2) -> (d0 * 2 + d1, d2)> by [<Unmerge{1, 2} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 2, 320] -> [2, 320]> : memref<2x320xf16> to memref<1x2x320xf16>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x1xf16>
    rock.reduce  sum %alloc_0 into %alloc_2 {axis = 1 : index, blockSize = 256 : i32, gridSize = 2 : i32} : memref<2x320xf16> into memref<2x1xf16>
    %9 = rock.transform %alloc_2 by <affine_map<(d0, d1, d2) -> (d0 * 2 + d1, d2)> by [<Unmerge{1, 2} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 2, 1] -> [2, 1]> : memref<2x1xf16> to memref<1x2x1xf16>
    memref.copy %8, %arg3 : memref<1x2x320xf16> to memref<1x2x320xf16>
    memref.copy %9, %arg4 : memref<1x2x1xf16> to memref<1x2x1xf16>
    return
  }
}
