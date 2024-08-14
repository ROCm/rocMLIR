// RUN: rocmlir-driver --rock-regularize %s | FileCheck %s
// RUN: rocmlir-driver --rock-regularize --rock-gridwise-gemm-to-blockwise --rock-blockwise-gemm-to-threadwise --rock-linalg-align --verify-passes %s | rocmlir-opt

func.func @rock_gemm(%arg0: memref<64x64x64xf16>, %arg1: memref<64x64x64xf32>, %arg2: memref<64x64x64xf32>, %arg3: memref<64x64x64xf32>) attributes {block_size = 256 : i32, grid_size = 64 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100", wave_size = 32 : i32} {
  %alloc = memref.alloc() : memref<64x64x64xf32>
  %arg0_tr = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG", "gemmK", "gemmM"] at [0, 1, 2] -> ["gemmG", "gemmK", "gemmM"] at [0, 1, 2]>] bounds = [64, 64, 64] -> [64, 64, 64]> : memref<64x64x64xf16> to memref<64x64x64xf16>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0_tr, %arg3 : memref<64x64x64xf16>, memref<64x64x64xf32>) outs(%alloc : memref<64x64x64xf32>) {
  ^bb0(%in: f16, %in_0: f32, %out: f32):
    %1 = arith.extf %in : f16 to f32
    %2 = arith.addf %1, %in_0 : f32
    linalg.yield %2 : f32
  }
  %0 = rock.transform %alloc by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [2, 1]>] bounds = [64, 64, 64] -> [64, 64, 64]> : memref<64x64x64xf32> to memref<64x64x64xf32>
  rock.gridwise_gemm %arg2 = %0 * %arg1 storeMethod(set) features =  dot|atomic_add|atomic_fmax_f32 {gridSize = 64 : i32, numCU = 48 : i32, params = #rock.general_gemm_params<blockSize = 256, kPerBlock = 16, mPerBlock = 64, nPerBlock = 64, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, splitKFactor = 1>} : memref<64x64x64xf32> = memref<64x64x64xf32> * memref<64x64x64xf32>
  return
}

// Annotate linalg.generic with rock.majorTensorNumber
// CHECK: linalg.generic
// CHECK-SAME: {rock.majorTensorNumber = 0
