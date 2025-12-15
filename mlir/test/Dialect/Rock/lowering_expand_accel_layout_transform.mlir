// RUN: rocmlir-driver -mlir-print-local-scope -rock-expand-accel-layout-transform -verify-passes %s | FileCheck %s

// CHECK-LABEL: @rock_gemm_both
func.func @rock_gemm_both(%arg0: memref<16384xf16>, %arg1: memref<16384xf16>, %arg2: memref<16384xf16>) {
  // CHECK-NOT: rock.accel_layout_transform %arg0
  // CHECK-NOT: rock.accel_layout_transform %arg1

  // CHECK-DAG: %[[trA0:.*]] = rock.transform %arg0
  // CHECK-DAG: %[[trA1:.*]] = rock.transform %[[trA0]]
  // CHECK-DAG: %[[trA2:.*]] = rock.transform %[[trA1]]

  // CHECK-DAG: %[[trB0:.*]] = rock.transform %arg1
  // CHECK-DAG: %[[trB1:.*]] = rock.transform %[[trB0]]
  // CHECK-DAG: %[[trB2:.*]] = rock.transform %[[trB1]]

  // CHECK-DAG: %[[trC0:.*]] = rock.transform %arg2
  
  // CHECK-DAG: rock.gemm %[[trC0]] = %[[trA2]] * %[[trB2]]

  %0 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %1 = rock.accel_layout_transform %arg0 {isA, params = #rock.mfma_gemm_params<kpackPerBlock = 2, mPerBlock = 32, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<16384xf16> to memref<1x128x128xf16>
  %2 = rock.accel_layout_transform %arg1 {params = #rock.mfma_gemm_params<kpackPerBlock = 2, mPerBlock = 32, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<16384xf16> to memref<1x128x128xf16>
  rock.gemm %0 = %1 * %2 features =  mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b storeMethod =  set {aAccelLayout, arch = "amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-", bAccelLayout, derivedBlockSize = 64 : i32, numCU = 304 : i32, params = #rock.mfma_gemm_params<kpackPerBlock = 2, mPerBlock = 32, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, wavesPerEU = 0, gridGroupSize = 0, outputSwizzle = 2, forceUnroll = true>, perf_config = "v3:32,32,2,32,32,8,1,2,2,1,1"} : memref<1x128x128xf16> = memref<1x128x128xf16> * memref<1x128x128xf16>
  return
}

// CHECK-LABEL: @rock_gemm_only_A
func.func @rock_gemm_only_A(%arg0: memref<16384xf16>, %arg1: memref<16384xf16>, %arg2: memref<16384xf16>) {
  // CHECK-NOT: rock.accel_layout_transform %arg0

  // CHECK-DAG: %[[trA0:.*]] = rock.transform %arg0
  // CHECK-DAG: %[[trA1:.*]] = rock.transform %[[trA0]]
  // CHECK-DAG: %[[trA2:.*]] = rock.transform %[[trA1]]

  // CHECK-DAG: %[[trB0:.*]] = rock.transform %arg1

  // CHECK-DAG: %[[trC0:.*]] = rock.transform %arg2
  
  // CHECK-DAG: rock.gemm %[[trC0]] = %[[trA2]] * %[[trB0]]

  %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %1 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %2 = rock.accel_layout_transform %arg0 {isA, params = #rock.mfma_gemm_params<kpackPerBlock = 2, mPerBlock = 32, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, wavesPerEU = 0, gridGroupSize = 0, outputSwizzle = 2, forceUnroll = true>} : memref<16384xf16> to memref<1x128x128xf16>
  rock.gemm %1 = %2 * %0 features =  mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b storeMethod =  set {aAccelLayout, arch = "amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-", derivedBlockSize = 64 : i32, numCU = 304 : i32, params = #rock.mfma_gemm_params<kpackPerBlock = 2, mPerBlock = 32, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, wavesPerEU = 0, gridGroupSize = 0, outputSwizzle = 2, forceUnroll = true>, perf_config = "v3:32,32,2,32,32,8,1,2,2,1,1"} : memref<1x128x128xf16> = memref<1x128x128xf16> * memref<1x128x128xf16>
  return
}

// CHECK-LABEL: @rock_gemm_only_B
func.func @rock_gemm_only_B(%arg0: memref<16384xf16>, %arg1: memref<16384xf16>, %arg2: memref<16384xf16>) {
  // CHECK-NOT: rock.accel_layout_transform %arg1

  // CHECK-DAG: %[[trA0:.*]] = rock.transform %arg0

  // CHECK-DAG: %[[trB0:.*]] = rock.transform %arg1
  // CHECK-DAG: %[[trB1:.*]] = rock.transform %[[trB0]]
  // CHECK-DAG: %[[trB2:.*]] = rock.transform %[[trB1]]

  // CHECK-DAG: %[[trC0:.*]] = rock.transform %arg2
  
  // CHECK-DAG: rock.gemm %[[trC0]] = %[[trA0]] * %[[trB2]]

  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %1 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %2 = rock.accel_layout_transform %arg1 {params = #rock.mfma_gemm_params<kpackPerBlock = 2, mPerBlock = 32, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, wavesPerEU = 0, gridGroupSize = 0, outputSwizzle = 2, forceUnroll = true>} : memref<16384xf16> to memref<1x128x128xf16>
  rock.gemm %1 = %0 * %2 features =  mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b storeMethod =  set {arch = "amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-", bAccelLayout, derivedBlockSize = 64 : i32, numCU = 304 : i32, params = #rock.mfma_gemm_params<kpackPerBlock = 2, mPerBlock = 32, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, wavesPerEU = 0, gridGroupSize = 0, outputSwizzle = 2, forceUnroll = true>, perf_config = "v3:32,32,2,32,32,8,1,2,2,1,1"} : memref<1x128x128xf16> = memref<1x128x128xf16> * memref<1x128x128xf16>
  return
}
