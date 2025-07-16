// RUN: not rocmlir-opt -rock-affix-params %s >/dev/null

func.func @mlir_dot_add(%arg0: memref<1x2x1024xf16>, %arg1: memref<1x1024x320xf16>, %arg2: memref<1x2x320xf16> ) attributes {arch = "gfx1201", kernel = "mixr", num_cu = 32 : i64} {
  rock.gemm %arg2 = %arg0 * %arg1 storeMethod =  set {perf_config = "v3:16,128,8,8,128,4,1,1,2,1,1"} : memref<1x2x320xf16> = memref<1x2x1024xf16> * memref<1x1024x320xf16>
  return
}
