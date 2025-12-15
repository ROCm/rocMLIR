// RUN: rocmlir-driver -mlir-print-local-scope -rock-affix-params -verify-passes %s | FileCheck %s

// CHECK-LABEL: @rock_gemm_accel
func.func @rock_gemm_accel(%arg0: memref<16384xf16>, %arg1: memref<16384xf16>, %arg2: memref<16384xf16>) {
  // CHECK: rock.accel_layout_transform %arg0
  // CHECK-SAME: params = #rock.mfma_gemm_params<kpackPerBlock = 2, mPerBlock = 32, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>
  // CHECK: rock.accel_layout_transform %arg1
  // CHECK-SAME: params = #rock.mfma_gemm_params<kpackPerBlock = 2, mPerBlock = 32, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.mfma_gemm_params<kpackPerBlock = 2, mPerBlock = 32, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>
  %0 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %1 = rock.accel_layout_transform %arg0 {isA} : memref<16384xf16> to memref<1x128x128xf16>
  %2 = rock.accel_layout_transform %arg1 : memref<16384xf16> to memref<1x128x128xf16>
  rock.gemm %0 = %1 * %2 features =  mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b storeMethod =  set {aAccelLayout, arch = "amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-", bAccelLayout, numCU = 304 : i32, perf_config = "v3:32,32,2,32,32,8,1,2,2,1,1"} : memref<1x128x128xf16> = memref<1x128x128xf16> * memref<1x128x128xf16>
  return
}
