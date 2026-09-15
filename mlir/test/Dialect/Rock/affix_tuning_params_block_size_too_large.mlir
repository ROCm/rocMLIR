// A workgroup larger than one of the global-to-LDS copy tiles leaves some
// threads with nothing to load, so rock-affix-params must reject the perf
// config instead of letting it fail later in gridwise-gemm-to-blockwise.

// Non-accel: kPerBlock * mPerBlock = 4 * 32 = 128 < blockSize = 256.
// RUN: not rocmlir-opt -rock-affix-params %s >/dev/null

func.func @general_gemm_block_size_too_large(%arg0: memref<1x32x2304xf32>, %arg1: memref<1x2304x512xf32>, %arg2: memref<1x32x512xf32>) attributes {rock.arch = "gfx1150", rock.kernel = "mixr", rock.num_cu = 6 : i64} {
  rock.gemm %arg2 = %arg0 * %arg1 storeMethod = set {perf_config = "v3:256,32,128,4,2,2,1,1,2"} : memref<1x32x512xf32> = memref<1x32x2304xf32> * memref<1x2304x512xf32>
  return
}
