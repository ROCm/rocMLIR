// RUN: rocmlir-opt -split-input-file -rock-gemm-to-gridwise -rock-regularize -rock-gridwise-gemm-to-blockwise %s | FileCheck %s

#xdlops_gemm_params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 128, kpack = 1, mPerWave = 64, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>
// CHECK-LABEL: @no_main_loop
func.func @no_main_loop(%arg0: memref<64xf32>, %arg1: memref<128xf32>, %arg2: memref<8192xf32>) attributes {block_size = 256 : i32, grid_size = 1 : i32} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 + d1) * 64 + d2)> by [<Unmerge{1, 1, 64} ["g", "k", "m"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [1, 1, 64] -> [64]> : memref<64xf32> to memref<1x1x64xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 + d1) * 128 + d2)> by [<Unmerge{1, 1, 128} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [1, 1, 128] -> [128]> : memref<128xf32> to memref<1x1x128xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 128 + d2)> by [<Unmerge{1, 64, 128} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [1, 64, 128] -> [8192]> : memref<8192xf32> to memref<1x64x128xf32>
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-NOT: rock.blockwise_gemm_accel
  rock.gemm %2 = tr %0 * %1 features =  mfma|dot|atomic_add storeMethod =  set {arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", derivedBlockSize = 256 : i32, params = #xdlops_gemm_params1} : memref<1x64x128xf32> = memref<1x1x64xf32> * memref<1x1x128xf32>
  return
}

// -----

#xdlops_gemm_params2 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 16, mPerBlock = 64, nPerBlock = 64, kpack = 4, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>
// CHECK-LABEL: @two_loops
func.func @two_loops(%arg0: memref<590592xf16>, %arg1: memref<590592xf16>, %arg2: memref<196608xf16>) attributes {block_size = 256 : i32, grid_size = 48 : i32} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 769 + d2)> by [<Unmerge{3, 256, 769} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 256, 769] -> [590592]> : memref<590592xf16> to memref<3x256x769xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 769 + d1) * 256 + d2)> by [<Unmerge{3, 769, 256} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 769, 256] -> [590592]> : memref<590592xf16> to memref<3x769x256xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 256 + d2)> by [<Unmerge{3, 256, 256} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 256, 256] -> [196608]> : memref<196608xf16> to memref<3x256x256xf16>
  %alloc = memref.alloc() : memref<3x256x256xf32>
  // CHECK-DAG: %[[ldsA:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[ldsB:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-DAG: rock.dealloc %[[ldsA]] : memref<8192xi8, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.dealloc %[[ldsB]] : memref<8192xi8, #gpu.address_space<workgroup>>

  // CHECK-DAG: %[[ldsA2:.+]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[ldsB2:.+]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-DAG: rock.dealloc %[[ldsA2]] : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.dealloc %[[ldsB2]] : memref<1024xi8, #gpu.address_space<workgroup>>
  rock.gemm %2 = %0 * %1 features =  mfma|dot|atomic_add storeMethod =  set {arch = "amdgcn-amd-amdhsa:gfx942", derivedBlockSize = 256 : i32, params = #xdlops_gemm_params2, perf_config = "v2:64,64,16,32,32,4,1,1,1"} : memref<3x256x256xf16> = memref<3x256x769xf16> * memref<3x769x256xf16>
  return
}

// -----


#xdlops_gemm_params3 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 16, mPerBlock = 64, nPerBlock = 64, kpack = 4, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>
// CHECK-LABEL: @reverse_grid
func.func @reverse_grid(%arg0: memref<590592xf16>, %arg1: memref<590592xf16>, %arg2: memref<196608xf16>) attributes {block_size = 256 : i32, grid_size = 48 : i32, reverse_grid} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 769 + d2)> by [<Unmerge{3, 256, 769} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 256, 769] -> [590592]> : memref<590592xf16> to memref<3x256x769xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 769 + d1) * 256 + d2)> by [<Unmerge{3, 769, 256} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 769, 256] -> [590592]> : memref<590592xf16> to memref<3x769x256xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 256 + d2)> by [<Unmerge{3, 256, 256} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 256, 256] -> [196608]> : memref<196608xf16> to memref<3x256x256xf16>
  // CHECK-DAG: %[[ldsA:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[ldsB:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-DAG: rock.dealloc %[[ldsA]] : memref<8192xi8, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.dealloc %[[ldsB]] : memref<8192xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.blockwise_gemm_accel
  rock.gemm %2 = %0 * %1 features =  mfma|dot|atomic_add storeMethod =  set {arch = "amdgcn-amd-amdhsa:gfx942", derivedBlockSize = 256 : i32, params = #xdlops_gemm_params3, perf_config = "v2:64,64,16,32,32,4,1,1,1"} : memref<3x256x256xf16> = memref<3x256x769xf16> * memref<3x769x256xf16>
  return
}