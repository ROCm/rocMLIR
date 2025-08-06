// RUN: rocmlir-gen -emit-module-fusibility-for=v3:16,16,4,16,16,1,5,1,2,1,1 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:1
// RUN: rocmlir-gen -emit-module-fusibility-for=v3:16,16,4,16,16,1,1,1,2,1,1 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
module {
  func.func @mlir_dot_add(%arg0: memref<1x2x320xf32>, %arg1: memref<1x2x320xf16>, %arg2: memref<1x2x1280xf32>, %arg3: memref<1x1280x320xf32>, %arg4: memref<1x2x320xf16>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x320xf32>
    rock.gemm %alloc = %arg2 * %arg3 storeMethod =  set : memref<1x2x320xf32> = memref<1x2x1280xf32> * memref<1x1280x320xf32>
    %0 = rock.transform %alloc by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 2} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [2, 320] -> [1, 2, 320]> : memref<1x2x320xf32> to memref<2x320xf32>
    %1 = rock.transform %arg0 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 2} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [2, 320] -> [1, 2, 320]> : memref<1x2x320xf32> to memref<2x320xf32>
    %2 = rock.transform %arg1 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 2} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [2, 320] -> [1, 2, 320]> : memref<1x2x320xf16> to memref<2x320xf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x320xf16>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %1, %2 : memref<2x320xf32>, memref<2x320xf32>, memref<2x320xf16>) outs(%alloc_0 : memref<2x320xf16>) {
    ^bb0(%in: f32, %in_1: f32, %in_2: f16, %out: f16):
      %3 = arith.addf %in, %in_1 : f32
      %4 = arith.extf %in_2 : f16 to f32
      %5 = arith.divf %in, %4 : f32
      %6 = arith.subf %3, %5 : f32
      %7 = arith.truncf %6 : f32 to f16
      linalg.yield %7 : f16
    }
    %8 = rock.transform %alloc_0 by <affine_map<(d0, d1, d2) -> (d0 * 2 + d1, d2)> by [<Unmerge{1, 2} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 2, 320] -> [2, 320]> : memref<2x320xf16> to memref<1x2x320xf16>
    memref.copy %8, %arg4 : memref<1x2x320xf16> to memref<1x2x320xf16>
    return
  }
}
