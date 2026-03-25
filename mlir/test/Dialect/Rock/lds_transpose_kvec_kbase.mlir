// RUN: rocmlir-opt -rock-gridwise-gemm-to-blockwise -split-input-file %s | FileCheck %s --check-prefix=KVEC_GE
// RUN: rocmlir-opt -rock-gridwise-gemm-to-blockwise -split-input-file %s | FileCheck %s --check-prefix=KVEC_LT

// Test AccelEmitter::wrapLDSBufferForLoad K-indexing formula
// For f16 with 32x32x16 MFMA: k_base = 8

// Case 1: kVec >= kBase (kpack=8, k_base=8) - single K iteration
#params_kvec_ge = #rock.accel_gemm_params<
  kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64,
  kpack = 8, mPerWave = 32, nPerWave = 32,
  mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 4,
  outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // KVEC_GE-LABEL: func.func @test_kvec_ge_kbase
  // KVEC_GE: scf.for %{{.*}} = %{{.*}} to %c1{{.*}} step
  // KVEC_GE: ldsTransposeEnabled = true
  func.func @test_kvec_ge_kbase(
      %arg0: memref<4096xf16>,
      %arg1: memref<4096xf16>,
      %arg2: memref<4096xf16>)
      attributes {block_size = 256 : i32, grid_size = 1 : i32,
                  enable_splitk_for_tuning, kernel,
                  num_cu = 256 : i64} {
    %a = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["k", "m"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf16> to memref<1x64x64xf16>
    %b = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf16> to memref<1x64x64xf16>
    %c = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf16> to memref<1x64x64xf16>

    rock.gridwise_gemm_accel(%a, %b, %c)
      storeMethod(set)
      features = mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b
      {blockSize = 256 : i32, gridSize = 1 : i32, params = #params_kvec_ge}
      : memref<1x64x64xf16>, memref<1x64x64xf16>, memref<1x64x64xf16>
    return
  }
}

// -----

// Case 2: kVec < kBase (kpack=4, k_base=8) - multiple K iterations (2)
#params_kvec_lt = #rock.accel_gemm_params<
  kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64,
  kpack = 4, mPerWave = 32, nPerWave = 32,
  mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 4,
  outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // KVEC_LT-LABEL: func.func @test_kvec_lt_kbase
  // KVEC_LT: scf.for %{{.*}} = %{{.*}} to %c2{{.*}} step
  // KVEC_LT: ldsTransposeEnabled = true
  func.func @test_kvec_lt_kbase(
      %arg0: memref<4096xf16>,
      %arg1: memref<4096xf16>,
      %arg2: memref<4096xf16>)
      attributes {block_size = 256 : i32, grid_size = 1 : i32,
                  enable_splitk_for_tuning, kernel,
                  num_cu = 256 : i64} {
    %a = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["k", "m"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf16> to memref<1x64x64xf16>
    %b = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf16> to memref<1x64x64xf16>
    %c = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf16> to memref<1x64x64xf16>

    rock.gridwise_gemm_accel(%a, %b, %c)
      storeMethod(set)
      features = mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b
      {blockSize = 256 : i32, gridSize = 1 : i32, params = #params_kvec_lt}
      : memref<1x64x64xf16>, memref<1x64x64xf16>, memref<1x64x64xf16>
    return
  }
}
