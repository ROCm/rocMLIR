// RUN: rocmlir-opt -rock-buffer-dependency-analysis %s | FileCheck %s


#map = affine_map<(d0, d1) -> (0, d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0 * 64 + d1, d2)>
#transform_map = #rock.transform_map<#map by [<Merge{1, 64} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [64, 64] -> [1, 64, 64]>
#transform_map1 = #rock.transform_map<#map2 by [<Unmerge{1, 64} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 64, 64] -> [64, 64]>


func.func @test1(%arg0: memref<1x64x1024xf16>, %arg1: memref<1x1024x64xf16>, %arg2: memref<1x64x64xf16>)
  attributes {arch = "gfx908:sramecc+:xnack-",
  kernel,
  expected = [{alloc_name = "alloc_0", writers = ["rock.gemm"], readers = ["linalg.generic"]},
              {alloc_name = "alloc_1", writers = ["linalg.generic"], readers = ["memref.copy"]}]} {
  %alloc_0 = memref.alloc() {alignment = 64 : i64, name = "alloc_0"} : memref<1x64x64xf16>
  rock.gemm %alloc_0 = %arg0 * %arg1 features =  mfma|dot|atomic_add storeMethod =  set {arch = "gfx908:sramecc+:xnack-"} : memref<1x64x64xf16> = memref<1x64x1024xf16> * memref<1x1024x64xf16>

  %0 = rock.transform %alloc_0 by #transform_map : memref<1x64x64xf16> to memref<64x64xf16>
  %alloc_1 = memref.alloc() {alignment = 64 : i64, name = "alloc_1"} : memref<64x64xf16>
  %cst = arith.constant 0.000000e+00 : f16
  linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<64x64xf16>) outs(%alloc_1 : memref<64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %2 = arith.maxf %in, %cst : f16
    linalg.yield %2 : f16
  }
  %1 = rock.transform %alloc_1 by #transform_map1 : memref<64x64xf16> to memref<1x64x64xf16>

  memref.copy %1, %arg2 : memref<1x64x64xf16> to memref<1x64x64xf16>
  return
}


func.func @test2(%arg0: memref<1x64x1024xf16>, %arg1: memref<1x1024x64xf16>, %arg2: memref<1x64x64xf16>, %arg3: memref<1x64x64xf16>) 
  attributes {arch = "gfx908:sramecc+:xnack-",
  kernel,
  expected = [{alloc_name = "alloc_0", writers = ["rock.gemm"], readers = ["memref.copy"]},
              {alloc_name = "alloc_1", writers = ["linalg.generic"], readers = ["memref.copy"]}]} {
  %alloc_0 = memref.alloc() {alignment = 64 : i64, name = "alloc_0"} : memref<1x64x64xf16>
  rock.gemm %alloc_0 = %arg0 * %arg1 features =  mfma|dot|atomic_add storeMethod =  set {arch = "gfx908:sramecc+:xnack-"} : memref<1x64x64xf16> = memref<1x64x1024xf16> * memref<1x1024x64xf16>
  
  %0 = rock.transform %arg3 by #transform_map : memref<1x64x64xf16> to memref<64x64xf16>
  %alloc_1 = memref.alloc() {alignment = 64 : i64, name = "alloc_1"} : memref<64x64xf16>
  %cst = arith.constant 0.000000e+00 : f16
  linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<64x64xf16>) outs(%alloc_1 : memref<64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %2 = arith.maxf %in, %cst : f16
    linalg.yield %2 : f16
  }
  %1 = rock.transform %alloc_1 by #transform_map1 : memref<64x64xf16> to memref<1x64x64xf16>

  memref.copy %alloc_0, %arg2 : memref<1x64x64xf16> to memref<1x64x64xf16>
  memref.copy %1, %arg3 : memref<1x64x64xf16> to memref<1x64x64xf16>
  return
}
