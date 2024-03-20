// RUN: rocmlir-opt -rock-gridwise-gemm-to-blockwise \
// RUN: --split-input-file --mlir-print-local-scope %s | FileCheck %s

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
  // CHECK-LABEL: @with_interchange
  // CHECK-SAME: ([[arg0:%.+]]: memref<1x64x64xf32>, [[arg1:%.+]]: memref<1x64x128xf32>, [[arg2:%.+]]: memref<1x128x64xf32>)
  // CHECK: [[trA:%.+]] = rock.transform [[arg0]]
  // CHECK: [[trC:%.+]] = rock.transform [[arg2]]
  // CHECK: [[newTrC:%.+]] = rock.transform [[trC]]
  // CHECK-SAME: by <affine_map<(d0, d1, d2) -> (d0, d2, d1)>
  // CHECK-SAME: by [<PassThrough ["gemmG", "gemmM", "gemmN"] at [0, 2, 1] -> ["gemmG", "gemmM", "gemmN"] at [0, 1, 2]>]
  // CHECK: rock.transform [[arg1]]
  // CHECK-SAME: "m_thread"
  // CHECK: rock.transform [[trA]]
  // CHECK-SAME: "n_thread"
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: #rock.xdlops_gemm_params<kpackPerBlock = 4, mPerBlock = 128, nPerBlock = 64, kpack = 4, mPerWave = 64, nPerWave = 32, forceUnroll = true>
  // CHECK: rock.threadwise_write_all {{.*}}([[newTrC]])
  func.func @with_interchange(%arg0: memref<1x64x64xf32>, %arg1: memref<1x64x128xf32>, %arg2: memref<1x128x64xf32>) attributes {block_size = 256 : i32, grid_size = 1 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %trA = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [2, 1]>] bounds = [1, 64, 64] -> [1, 64, 64]> : memref<1x64x64xf32> to memref<1x64x64xf32>
    %trC = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM", "gemmN"] at [1, 2] -> ["gemmM", "gemmN"] at [2, 1]>] bounds = [1, 64, 128] -> [1, 128, 64]> : memref<1x128x64xf32> to memref<1x64x128xf32>
    rock.gridwise_gemm_accel(%trA, %arg1, %trC) storeMethod( set) features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", blockSize = 256 : i32, gridSize = 1 : i32, numCU = 104 : i32, params = #rock.xdlops_gemm_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 128, kpack = 4, mPerWave = 32, nPerWave = 64, forceUnroll = true>} : memref<1x64x64xf32>, memref<1x64x128xf32>, memref<1x64x128xf32>
    return
  }

  // CHECK-LABEL: @without_interchange
  // CHECK-SAME: ([[arg0:%.+]]: memref<1x64x64xf32>, [[arg1:%.+]]: memref<1x64x128xf32>, [[arg2:%.+]]: memref<1x64x128xf32>)
  // CHECK: [[trA:%.+]] = rock.transform [[arg0]]
  // CHECK: rock.transform [[trA]]
  // CHECK-SAME: "m_thread"
  // CHECK: rock.transform [[arg1]]
  // CHECK-SAME: "n_thread"
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: #rock.xdlops_gemm_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 128, kpack = 4, mPerWave = 32, nPerWave = 64, forceUnroll = true>
  // CHECK: rock.threadwise_write_all {{.*}}([[arg2]])
  // CHECK-SAME: memref<1x64x128xf32>
  func.func @without_interchange(%arg0: memref<1x64x64xf32>, %arg1: memref<1x64x128xf32>, %arg2: memref<1x64x128xf32>) attributes {block_size = 256 : i32, grid_size = 1 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %trA = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [2, 1]>] bounds = [1, 64, 64] -> [1, 64, 64]> : memref<1x64x64xf32> to memref<1x64x64xf32>
    rock.gridwise_gemm_accel(%trA, %arg1, %arg2) storeMethod( set) features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", blockSize = 256 : i32, gridSize = 1 : i32, numCU = 104 : i32, params = #rock.xdlops_gemm_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 128, kpack = 4, mPerWave = 32, nPerWave = 64, forceUnroll = true>} : memref<1x64x64xf32>, memref<1x64x128xf32>, memref<1x64x128xf32>
    return
  }

  // CHECK-LABEL: @transpose_after_fusion
  // CHECK-SAME: ([[arg0:%.+]]: memref<1x64x64xf32>, [[arg1:%.+]]: memref<1x64x128xf32>, [[arg2:%.+]]: memref<1x128x64xf32>)
  // CHECK: [[gemmOut:%.+]] = memref.alloc
  // CHECK: [[trA:%.+]] = rock.transform [[arg0]]
  // CHECK: [[trC:%.+]] = rock.transform [[arg2]]
  // CHECK: [[newTrC:%.+]] = rock.transform [[gemmOut]]
  // CHECK-SAME: by <affine_map<(d0, d1, d2) -> (d0, d2, d1)>
  // CHECK-SAME: by [<PassThrough ["gemmG", "gemmM", "gemmN"] at [0, 2, 1] -> ["gemmG", "gemmM", "gemmN"] at [0, 1, 2]>]
  // CHECK: rock.transform [[arg1]]
  // CHECK-SAME: "m_thread"
  // CHECK: rock.transform [[trA]]
  // CHECK-SAME: "n_thread"
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: #rock.xdlops_gemm_params<kpackPerBlock = 4, mPerBlock = 128, nPerBlock = 64, kpack = 4, mPerWave = 64, nPerWave = 32, forceUnroll = true>
  // CHECK: rock.threadwise_write_all {{.*}}([[newTrC]])
  func.func @transpose_after_fusion(%arg0: memref<1x64x64xf32>, %arg1: memref<1x64x128xf32>, %arg2: memref<1x128x64xf32>) attributes {block_size = 256 : i32, grid_size = 1 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %gemmOut = memref.alloc() : memref<1x64x128xf32>
    %trA = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [2, 1]>] bounds = [1, 64, 64] -> [1, 64, 64]> : memref<1x64x64xf32> to memref<1x64x64xf32>
    %trC = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM", "gemmN"] at [1, 2] -> ["gemmM", "gemmN"] at [2, 1]>] bounds = [1, 64, 128] -> [1, 128, 64]> : memref<1x128x64xf32> to memref<1x64x128xf32>
    rock.gridwise_gemm_accel(%trA, %arg1, %gemmOut) storeMethod( set) features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", blockSize = 256 : i32, gridSize = 1 : i32, numCU = 104 : i32, params = #rock.xdlops_gemm_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 128, kpack = 4, mPerWave = 32, nPerWave = 64, forceUnroll = true>} : memref<1x64x64xf32>, memref<1x64x128xf32>, memref<1x64x128xf32>
    // Note: This is a substitute for a real fusion.
    memref.copy %gemmOut, %trC : memref<1x64x128xf32> to memref<1x64x128xf32>
    return
  }

  // CHECK-LABEL: @no_interchange_after_complex_fusion
  // CHECK-SAME: ([[arg0:%.+]]: memref<1x64x64xf32>, [[arg1:%.+]]: memref<1x64x128xf32>, [[arg2:%.+]]: memref<1x128x64xf32>, [[arg3:%.+]]: memref<1x64x128xf32>)
  // CHECK: [[gemmOut:%.+]] = memref.alloc
  // CHECK: [[trA:%.+]] = rock.transform [[arg0]]
  // CHECK: rock.transform [[trA]]
  // CHECK-SAME: "m_thread"
  // CHECK: rock.transform [[arg1]]
  // CHECK-SAME: "n_thread"
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: #rock.xdlops_gemm_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 128, kpack = 4, mPerWave = 32, nPerWave = 64, forceUnroll = true>
  // CHECK: rock.threadwise_write_all {{.*}}([[gemmOut]])
  func.func @no_interchange_after_complex_fusion(%arg0: memref<1x64x64xf32>, %arg1: memref<1x64x128xf32>, %arg2: memref<1x128x64xf32>, %arg3: memref<1x64x128xf32>) attributes {block_size = 256 : i32, grid_size = 1 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %gemmOut = memref.alloc() : memref<1x64x128xf32>
    %trA = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [2, 1]>] bounds = [1, 64, 64] -> [1, 64, 64]> : memref<1x64x64xf32> to memref<1x64x64xf32>
    %trC = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM", "gemmN"] at [1, 2] -> ["gemmM", "gemmN"] at [2, 1]>] bounds = [1, 64, 128] -> [1, 128, 64]> : memref<1x128x64xf32> to memref<1x64x128xf32>
    rock.gridwise_gemm_accel(%trA, %arg1, %gemmOut) storeMethod( set) features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", blockSize = 256 : i32, gridSize = 1 : i32, numCU = 104 : i32, params = #rock.xdlops_gemm_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 128, kpack = 4, mPerWave = 32, nPerWave = 64, forceUnroll = true>} : memref<1x64x64xf32>, memref<1x64x128xf32>, memref<1x64x128xf32>
    // Note: This is a substitute for some suitably complex post-gemm graph.
    memref.copy %gemmOut, %trC : memref<1x64x128xf32> to memref<1x64x128xf32>
    memref.copy %gemmOut, %arg3 : memref<1x64x128xf32> to memref<1x64x128xf32>
    return
  }

  // CHECK-LABEL: @no_interchange_non_symmetric_perf_config
  // CHECK-SAME: ([[arg0:%.+]]: memref<1x64x64xf32>, [[arg1:%.+]]: memref<1x64x128xf32>, [[arg2:%.+]]: memref<1x128x64xf32>)
  // CHECK: [[trA:%.+]] = rock.transform [[arg0]]
  // CHECK: [[trC:%.+]] = rock.transform [[arg2]]
  // CHECK: rock.transform [[trA]]
  // CHECK-SAME: "m_thread"
  // CHECK: rock.transform [[arg1]]
  // CHECK-SAME: "n_thread"
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: #rock.xdlops_gemm_params<kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 64, kpack = 4, mPerWave = 4, nPerWave = 64, forceUnroll = true>
  // CHECK: rock.threadwise_write_all {{.*}}([[trC]])
  func.func @no_interchange_non_symmetric_perf_config(%arg0: memref<1x64x64xf32>, %arg1: memref<1x64x128xf32>, %arg2: memref<1x128x64xf32>) attributes {block_size = 256 : i32, grid_size = 8 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %trA = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [2, 1]>] bounds = [1, 64, 64] -> [1, 64, 64]> : memref<1x64x64xf32> to memref<1x64x64xf32>
    %trC = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM", "gemmN"] at [1, 2] -> ["gemmM", "gemmN"] at [2, 1]>] bounds = [1, 64, 128] -> [1, 128, 64]> : memref<1x128x64xf32> to memref<1x64x128xf32>
    rock.gridwise_gemm_accel(%trA, %arg1, %trC) storeMethod( set) features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", blockSize = 256 : i32, gridSize = 8 : i32, numCU = 104 : i32, params = #rock.xdlops_gemm_params<kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 64, kpack = 4, mPerWave = 4, nPerWave = 64, forceUnroll = true>} : memref<1x64x64xf32>, memref<1x64x128xf32>, memref<1x64x128xf32>
    return
  }
}

// -----

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx1030"} {
  // CHECK-LABEL: @general_gemm_interchange
  // CHECK-SAME: ([[arg0:%.+]]: memref<1x64x64xf32>, [[arg1:%.+]]: memref<1x64x128xf32>, [[arg2:%.+]]: memref<1x128x64xf32>)
  // CHECK: [[trA:%.+]] = rock.transform [[arg0]]
  // CHECK: [[trC:%.+]] = rock.transform [[arg2]]
  // CHECK: [[newTrC:%.+]] = rock.transform [[trC]]
  // CHECK-SAME: by <affine_map<(d0, d1, d2) -> (d0, d2, d1)>
  // CHECK-SAME: by [<PassThrough ["gemmG", "gemmM", "gemmN"] at [0, 2, 1] -> ["gemmG", "gemmM", "gemmN"] at [0, 1, 2]>]
  // CHECK: rock.transform [[arg1]]
  // CHECK-SAME: "m_thread"
  // CHECK: rock.transform [[trA]]
  // CHECK-SAME: "n_thread"
  // CHECK: rock.blockwise_gemm
  // CHECK-SAME: #rock.general_gemm_params<blockSize = 128, kPerBlock = 16, mPerBlock = 64, nPerBlock = 32, kPerThread = 1, mPerThread = 4, nPerThread = 2, kpack = 1>
  // CHECK: rock.threadwise_write_all {{.*}}([[newTrC]])
  func.func @general_gemm_interchange(%arg0: memref<1x64x64xf32>, %arg1: memref<1x64x128xf32>, %arg2: memref<1x128x64xf32>) attributes {block_size = 128 : i32, grid_size = 4 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1030"} {
    %trA = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [2, 1]>] bounds = [1, 64, 64] -> [1, 64, 64]> : memref<1x64x64xf32> to memref<1x64x64xf32>
    %trC = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM", "gemmN"] at [1, 2] -> ["gemmM", "gemmN"] at [2, 1]>] bounds = [1, 64, 128] -> [1, 128, 64]> : memref<1x128x64xf32> to memref<1x64x128xf32>
    rock.gridwise_gemm %trC = %trA * %arg1 features =  dot|atomic_fmax_f32 {gridSize = 4 : i32, numCU = 36 : i32, params = #rock.general_gemm_params<blockSize = 128, kPerBlock = 16, mPerBlock = 32, nPerBlock = 64, kPerThread = 1, mPerThread = 2, nPerThread = 4, kpack = 1>} : memref<1x64x128xf32> = memref<1x64x64xf32> * memref<1x64x128xf32>
    return
  }
}

// -----

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK-LABEL: @wmma_interchange
  // CHECK-SAME: ([[arg0:%.+]]: memref<1x64x64xf16>, [[arg1:%.+]]: memref<1x64x128xf16>, [[arg2:%.+]]: memref<1x128x64xf32>)
  // CHECK: [[trA:%.+]] = rock.transform [[arg0]]
  // CHECK: [[trC:%.+]] = rock.transform [[arg2]]
  // CHECK: [[newTrC:%.+]] = rock.transform [[trC]]
  // CHECK-SAME: by <affine_map<(d0, d1, d2) -> (d0, d2, d1)>
  // CHECK-SAME: by [<PassThrough ["gemmG", "gemmM", "gemmN"] at [0, 2, 1] -> ["gemmG", "gemmM", "gemmN"] at [0, 1, 2]>]
  // CHECK: rock.transform [[arg1]]
  // CHECK-SAME: "m_thread"
  // CHECK: rock.transform [[trA]]
  // CHECK-SAME: "n_thread"
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: #rock.wmma_gemm_params<kpackPerBlock = 4, mPerBlock = 128, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 64, forceUnroll = true>
  // CHECK: rock.threadwise_write_all {{.*}}([[newTrC]])
  func.func @wmma_interchange(%arg0: memref<1x64x64xf16>, %arg1: memref<1x64x128xf16>, %arg2: memref<1x128x64xf32>) attributes {block_size = 128 : i32, grid_size = 1 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
    %trA = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [2, 1]>] bounds = [1, 64, 64] -> [1, 64, 64]> : memref<1x64x64xf16> to memref<1x64x64xf16>
    %trC = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM", "gemmN"] at [1, 2] -> ["gemmM", "gemmN"] at [2, 1]>] bounds = [1, 64, 128] -> [1, 128, 64]> : memref<1x128x64xf32> to memref<1x64x128xf32>
    rock.gridwise_gemm_accel(%trA, %arg1, %trC) storeMethod( set) features =  dot|atomic_add|atomic_fmax_f32|wmma {arch = "amdgcn-amd-amdhsa:gfx1100", blockSize = 128 : i32, gridSize = 1 : i32, numCU = 48 : i32, params = #rock.wmma_gemm_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 128, kpack = 8, mPerWave = 64, nPerWave = 32, forceUnroll = true>} : memref<1x64x64xf16>, memref<1x64x128xf16>, memref<1x64x128xf32>
    return
  }
}
