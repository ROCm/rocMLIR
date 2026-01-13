// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise %s | FileCheck %s

#params_16 = #rock.accel_gemm_params<
  kpackPerBlock = 32, mPerBlock = 256, nPerBlock = 256,
  kpack = 1, mPerWave = 64, nPerWave = 64,
  mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 4,
  outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>

#params_8 = #rock.accel_gemm_params<
  kpackPerBlock = 32, mPerBlock = 256, nPerBlock = 128,
  kpack = 1, mPerWave = 64, nPerWave = 64,
  mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 4,
  outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK-LABEL: func.func @test_16_waves
  func.func @test_16_waves(
      %arg0: memref<2048xf16>,
      %arg1: memref<2048xf16>,
      %arg2: memref<4096xf16>)
      attributes {block_size = 1024 : i32, grid_size = 1 : i32,
                  enable_splitk_for_tuning, kernel,
                  num_cu = 256 : i64} {
    %a = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{32, 64} ["k", "m"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 32, 64] -> [2048]> : memref<2048xf16> to memref<1x32x64xf16>
    %b = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{32, 64} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 32, 64] -> [2048]> : memref<2048xf16> to memref<1x32x64xf16>
    %c = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf16> to memref<1x64x64xf16>

    // CHECK-NOT: ldsTransposeEnabled
    rock.gridwise_gemm_accel(%a, %b, %c)
      storeMethod(set)
      {blockSize = 1024 : i32, gridSize = 1 : i32, params = #params_16}
      : memref<1x32x64xf16>, memref<1x32x64xf16>, memref<1x64x64xf16>
    return
  }

  // CHECK-LABEL: func.func @test_8_waves
  func.func @test_8_waves(
      %arg0: memref<2048xf16>,
      %arg1: memref<2048xf16>,
      %arg2: memref<4096xf16>)
      attributes {block_size = 512 : i32, grid_size = 1 : i32,
                  enable_splitk_for_tuning, kernel,
                  num_cu = 256 : i64} {
    %a = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{32, 64} ["k", "m"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 32, 64] -> [2048]> : memref<2048xf16> to memref<1x32x64xf16>
    %b = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{32, 64} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 32, 64] -> [2048]> : memref<2048xf16> to memref<1x32x64xf16>
    %c = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf16> to memref<1x64x64xf16>

    // CHECK-NOT: ldsTransposeEnabled
    rock.gridwise_gemm_accel(%a, %b, %c)
      storeMethod(set)
      {blockSize = 512 : i32, gridSize = 1 : i32, params = #params_8}
      : memref<1x32x64xf16>, memref<1x32x64xf16>, memref<1x64x64xf16>
    return
  }
}
