// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -rock-blockwise-load-tile-to-threadwise -rock-blockwise-gemm-to-threadwise %s | FileCheck %s

#params = #rock.mfma_gemm_params<
  kpackPerBlock = 16, mPerBlock = 64, nPerBlock = 64,
  kpack = 1, mPerWave = 32, nPerWave = 32,
  mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 3,
  outputSwizzle = 2, forceUnroll = true>

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK-LABEL: func.func @test_lds_transpose_attributes
  func.func @test_lds_transpose_attributes(
      %arg0: memref<1024xf16>,
      %arg1: memref<1024xf16>,
      %arg2: memref<4096xf16>)
      attributes {block_size = 256 : i32, grid_size = 1 : i32,
                  enable_splitk_for_tuning, kernel,
                  num_cu = 256 : i64} {
    %a = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{16, 64} ["k", "m"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 16, 64] -> [1024]> : memref<1024xf16> to memref<1x16x64xf16>
    %b = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{16, 64} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 16, 64] -> [1024]> : memref<1024xf16> to memref<1x16x64xf16>
    %c = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf16> to memref<1x64x64xf16>

    rock.gridwise_gemm_accel(%a, %b, %c)
      storeMethod(set)
      features = mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b
      {blockSize = 256 : i32, gridSize = 1 : i32, params = #params}
      : memref<1x16x64xf16>, memref<1x16x64xf16>, memref<1x64x64xf16>
    return
  }
}

// CHECK: rock.threadwise_read_into {forceUnroll, ldsTransposeConfig = #rock.lds_transpose_config<dDim = 32, kDim = 16, mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, mPerWave = 32, nPerWave = 32, doubleBuffering = false, isOperandA = true>, useIndexDiffs} [](%{{.*}}) [%{{.*}}, %{{.*}}] -> %{{.*}} : memref<256x1x8xf16, #gpu.address_space<workgroup>> -> memref<8xf16, #gpu.address_space<private>>
// CHECK: rock.threadwise_read_into {forceUnroll, ldsTransposeConfig = #rock.lds_transpose_config<dDim = 32, kDim = 16, mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, mPerWave = 32, nPerWave = 32, doubleBuffering = false, isOperandA = false>, useIndexDiffs} [](%{{.*}}) [%{{.*}}, %{{.*}}] -> %{{.*}} : memref<256x1x8xf16, #gpu.address_space<workgroup>> -> memref<8xf16, #gpu.address_space<private>>

// -----

#params_double = #rock.mfma_gemm_params<
  kpackPerBlock = 32, mPerBlock = 64, nPerBlock = 64,
  kpack = 1, mPerWave = 16, nPerWave = 64,
  mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 4,
  outputSwizzle = 2, forceUnroll = true>

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK-LABEL: func.func @test_lds_transpose_attributes_double_buffering
  func.func @test_lds_transpose_attributes_double_buffering(
      %arg0: memref<2048xf16>,
      %arg1: memref<2048xf16>,
      %arg2: memref<4096xf16>)
      attributes {block_size = 256 : i32, grid_size = 1 : i32,
                  enable_splitk_for_tuning, kernel,
                  num_cu = 256 : i64} {
    %a = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{32, 64} ["k", "m"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 32, 64] -> [2048]> : memref<2048xf16> to memref<1x32x64xf16>
    %b = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{32, 64} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 32, 64] -> [2048]> : memref<2048xf16> to memref<1x32x64xf16>
    %c = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf16> to memref<1x64x64xf16>

    rock.gridwise_gemm_accel(%a, %b, %c)
      storeMethod(set)
      features = mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b
      {blockSize = 256 : i32, gridSize = 1 : i32, params = #params_double}
      : memref<1x32x64xf16>, memref<1x32x64xf16>, memref<1x64x64xf16>
    return
  }
}

// CHECK: rock.threadwise_read_into {forceUnroll, ldsTransposeConfig = #rock.lds_transpose_config<dDim = 16, kDim = 32, mPerBlock = 64, nPerBlock = 64, kPerBlock = 32, mPerWave = 16, nPerWave = 64, doubleBuffering = true, isOperandA = true>, useIndexDiffs}
// CHECK-SAME: memref<256x8xf16, #gpu.address_space<workgroup>> -> memref<8xf16, #gpu.address_space<private>>
// CHECK: rock.threadwise_read_into {forceUnroll, ldsTransposeConfig = #rock.lds_transpose_config<dDim = 16, kDim = 32, mPerBlock = 64, nPerBlock = 64, kPerBlock = 32, mPerWave = 16, nPerWave = 64, doubleBuffering = true, isOperandA = false>, useIndexDiffs}
// CHECK-SAME: memref<256x32xf16, #gpu.address_space<workgroup>> -> memref<32xf16, #gpu.address_space<private>>
