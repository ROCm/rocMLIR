// RUN: rocmlir-opt --rock-fold-broadcast %s | FileCheck %s
#map = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0 * 16 + d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["dim2", "dim0", "dim1"] at [0, 1, 2] -> ["dim2", "dim0", "dim1"] at [2, 0, 1]>] bounds = [1, 8, 32] -> [8, 32, 1]>
#transform_map1 = #rock.transform_map<#map1 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [4, 8, 32] -> [1, 8, 32]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{1, 16} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 16, 32] -> [16, 32]>
#transform_map3 = #rock.transform_map<#map1 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [4, 16, 32] -> [1, 16, 32]>
module {
  func.func @mlir_dot_add(%arg0: memref<8x32x1xf16>, %arg1: memref<4x8x16xf16>, %arg2: memref<16x32xf16>, %arg3: memref<4x8x32xf16>) attributes {arch = "", kernel} {
    %0 = rock.transform %arg0 by #transform_map : memref<8x32x1xf16> to memref<1x8x32xf16>
    %1 = rock.transform %0 by #transform_map1 : memref<1x8x32xf16> to memref<4x8x32xf16>
    %2 = rock.transform %arg2 by #transform_map2 : memref<16x32xf16> to memref<1x16x32xf16>
    %3 = rock.transform %2 by #transform_map3 : memref<1x16x32xf16> to memref<4x16x32xf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x8x32xf16>
    // CHECK: %[[foldA:.*]] = rock.transform %arg1 by {{.*}} : memref<4x8x16xf16> to memref<32x16xf16>
    // CHECK: %[[unbroadcastB:.*]] = rock.transform %3 by {{.*}} : memref<4x16x32xf16> to memref<16x32xf16>
    // CHECK: %[[foldC:.*]] = rock.transform %alloc by {{.*}} : memref<4x8x32xf16> to memref<32x32xf16>
    // CHECK: rock.gemm %[[foldC]] = %[[foldA]] * %[[unbroadcastB]] {{.*}} : memref<32x32xf16> = memref<32x16xf16> * memref<16x32xf16>
    rock.gemm %alloc = %arg1 * %3 features =  none storeMethod =  set {arch = ""} : memref<4x8x32xf16> = memref<4x8x16xf16> * memref<4x16x32xf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32xf16>
    linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc, %1 : memref<4x8x32xf16>, memref<4x8x32xf16>) outs(%alloc_0 : memref<4x8x32xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %4 = arith.addf %in, %in_1 : f16
      linalg.yield %4 : f16
    }
    memref.copy %alloc_0, %arg3 : memref<4x8x32xf16> to memref<4x8x32xf16>
    return
  }
}
