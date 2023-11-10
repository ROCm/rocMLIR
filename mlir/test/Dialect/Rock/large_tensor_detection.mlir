// RUN: rocmlir-opt %s -split-input-file -rock-gemm-to-gridwise \
// RUN:    -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering \
// RUN: | FileCheck %s --check-prefixes=BLOCKWISE
// RUN: rocmlir-opt %s -split-input-file -rock-kernel-pipeline | FileCheck %s --check-prefixes=GPU

// Arbitrary testcase: the tuning parameters are set to prevent needing to go
// through `-rock-affix-params` and can be replaced as needed.
#general_gemm_params = #rock.general_gemm_params<block_size = 64, kPerBlock = 16, mPerBlock = 64, nPerBlock = 32, kPerThread = 1, mPerThread = 4, nPerThread = 2, kpack = 1>
module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
// BLOCKWISE-LABEL: @rock_gemm
// BLOCKWISE-SAME: rock.is_64_bit_index
// GPU-LABEL: @rock_gemm_module
// GPU-SAME: dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64 : i32>>
  func.func @rock_gemm(%arg0: memref<1x32768x32768xf32>, %arg1: memref<1x32768x1xf32>, %arg2: memref<1x32768x1xf32>) attributes {rock.block_size = 64 : i32, rock.grid_size = 512 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100", rock.wave_size = 32 : i32} {
    rock.gemm %arg2 = %arg0 * %arg1 features =  dot|atomic_add|atomic_fmax_f32 storeMethod =  set {arch = "amdgcn-amd-amdhsa:gfx1100", grid_size = 512 : i32, params = #general_gemm_params} : memref<1x32768x1xf32> = memref<1x32768x32768xf32> * memref<1x32768x1xf32>
    return
  }
}

// -----

#general_gemm_params = #rock.general_gemm_params<block_size = 64, kPerBlock = 16, mPerBlock = 64, nPerBlock = 32, kPerThread = 1, mPerThread = 4, nPerThread = 2, kpack = 1>
module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
// BLOCKWISE-LABEL: @rock_gemm
// BLOCKWISE-NOT: rock.is_64_bit_index
// GPU-LABEL: @rock_gemm_module
// GPU-SAME: dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>
  func.func @rock_gemm(%arg0: memref<1x8192x8192xf32>, %arg1: memref<1x8192x1xf32>, %arg2: memref<1x8192x1xf32>) attributes {rock.block_size = 64 : i32, rock.grid_size = 128 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100", rock.wave_size = 32 : i32} {
    rock.gemm %arg2 = %arg0 * %arg1 features =  dot|atomic_add|atomic_fmax_f32 storeMethod =  set {arch = "amdgcn-amd-amdhsa:gfx1100", grid_size = 128 : i32, params = #general_gemm_params} : memref<1x8192x1xf32> = memref<1x8192x8192xf32> * memref<1x8192x1xf32>
    return
  }
}
