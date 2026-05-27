// RUN: rocmlir-opt -rock-expand-strides-lowering %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> ((d0 * 16 + d1) * 24 + d2)>
#map1 = affine_map<(d0, d1, d2) -> ((d0 * 24 + d1) * 16 + d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0) -> (d0 floordiv 1152, (d0 mod 1152) floordiv 24, d0 mod 24)>
#transform_map = #rock.transform_map<#map by [<Unmerge{4, 16, 24} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [4, 16, 24] -> [1536]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{4, 24, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [4, 24, 16] -> [1536]>
#transform_map2 = #rock.transform_map<#map3 by [<Merge{4, 48, 24} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [4608] -> [4, 48, 24]>
// CHECK-DAG: = #rock.transform_map<{{.*}}Slice{{.*}}0, 4, 0, 5, 0, 24{{.*}}bounds = [4, 5, 24] -> [4, 12, 24]>

module {
  // CHECK-LABEL: func.func @expand_strides_basic
  func.func @expand_strides_basic(%arg0: memref<1536xf16>, %arg1: memref<1536xf16>, %arg2: memref<4608xf16>) attributes {rock.arch = "gfx950", rock.kernel = "mixr"} {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = rock.transform %arg1 by #transform_map : memref<1536xf16> to memref<4x16x24xf16>
    %1 = rock.transform %arg0 by #transform_map1 : memref<1536xf16> to memref<4x24x16xf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x24x24xf16>
    rock.gemm %alloc = %1 * %0 storeMethod =  set : memref<4x24x24xf16> = memref<4x24x16xf16> * memref<4x16x24xf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<4x24x24xf16>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<4x24x24xf16>) outs(%alloc_0 : memref<4x24x24xf16>) {
    ^bb0(%in: f16, %out: f16):
      %3 = arith.negf %in : f16
      %4 = math.exp %3 : f16
      %5 = arith.addf %4, %cst : f16
      %6 = arith.divf %cst, %5 : f16
      linalg.yield %6 : f16
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x48x24xf16>
    rock.expand_strides %alloc_0 into %alloc_1 : memref<4x24x24xf16> into memref<4x48x24xf16>
    // CHECK: %[[TRANSFORM:.*]] = rock.transform %alloc_1 {{.*}} memref<4x48x24xf16> to memref<4x24x24xf16>
    // CHECK: memref.copy %alloc_0, %[[TRANSFORM]] : memref<4x24x24xf16> to memref<4x24x24xf16>
    %2 = rock.transform %alloc_1 by #transform_map2 : memref<4x48x24xf16> to memref<4608xf16>
    memref.copy %2, %arg2 : memref<4608xf16> to memref<4608xf16>
    return
  }

  // CHECK-LABEL: func.func @expand_strides_non_multiple
  func.func @expand_strides_non_multiple(%arg0: memref<320xf16>, %arg1: memref<1536xf16>, %arg2: memref<1152xf16>) attributes {rock.arch = "gfx1201", rock.kernel = "mixr", rock.num_cu = 32 : i64} {
  %cst = arith.constant 1.000000e+00 : f16
  %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 16 + d1) * 24 + d2)> by [<Unmerge{4, 16, 24} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [4, 16, 24] -> [1536]> : memref<1536xf16> to memref<4x16x24xf16>
  %1 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 5 + d1) * 16 + d2)> by [<Unmerge{4, 5, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [4, 5, 16] -> [320]> : memref<320xf16> to memref<4x5x16xf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x5x24xf16>
  rock.gemm %alloc = %1 * %0 storeMethod =  set : memref<4x5x24xf16> = memref<4x5x16xf16> * memref<4x16x24xf16>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<4x5x24xf16>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<4x5x24xf16>) outs(%alloc_0 : memref<4x5x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %3 = arith.negf %in : f16
    %4 = math.exp %3 : f16
    %5 = arith.addf %4, %cst : f16
    %6 = arith.divf %cst, %5 : f16
    linalg.yield %6 : f16
  }
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x12x24xf16>
  // CHECK: %[[TRANSFORM2:.*]] = rock.transform %alloc_1 {{.*}} : memref<4x12x24xf16> to memref<4x5x24xf16>
  // CHECK: memref.copy %alloc_0, %[[TRANSFORM2]] : memref<4x5x24xf16> to memref<4x5x24xf16>
  rock.expand_strides %alloc_0 into %alloc_1 : memref<4x5x24xf16> into memref<4x12x24xf16>
  %2 = rock.transform %alloc_1 by <affine_map<(d0) -> (d0 floordiv 288, (d0 mod 288) floordiv 24, d0 mod 24)> by [<Merge{4, 12, 24} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [1152] -> [4, 12, 24]> : memref<4x12x24xf16> to memref<1152xf16>
  memref.copy %2, %arg2 : memref<1152xf16> to memref<1152xf16>
  return
}
}

