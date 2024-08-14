// RUN: rocmlir-opt --rock-regularize --mlir-print-local-scope %s | FileCheck %s
// RUN: rocmlir-driver --kernel-pipeline=gpu --arch "gfx908:sramecc+:xnack-"

// -----// IR Dump After RockGemmToGridwisePass (rock-gemm-to-gridwise) //----- //
// CHECK-LABEL: @mlir_dot_mul
// CHECK-COUNT-4: memref.copy
// CHECK-NOT: memref.copy
// CHECK: return
func.func @mlir_dot_mul(%arg0: memref<6xf32>, %arg1: memref<12xf32>, %arg2: memref<8xf32>, %arg3: memref<8xf32>) attributes {arch = "gfx908:sramecc+:xnack-", block_size = 64 : i32, grid_size = 1 : i32, kernel = "mixr", num_cu = 120 : i64} {
  %cst = arith.constant 2.500000e-01 : f32
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 2 + d1) * 3 + d2)> by [<Unmerge{1, 2, 3} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 2, 3] -> [6]> : memref<6xf32> to memref<1x2x3xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 3 + d1) * 4 + d2)> by [<Unmerge{1, 3, 4} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 3, 4] -> [12]> : memref<12xf32> to memref<1x3x4xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x4xf32>
  %2 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [2, 1]>] bounds = [1, 3, 2] -> [1, 2, 3]> : memref<1x2x3xf32> to memref<1x3x2xf32>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 13} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 14} ["gemmMPad"] at [2] -> ["gemmM"] at [2]>] bounds = [1, 16, 16] -> [1, 3, 2]> : memref<1x3x2xf32> to memref<1x16x16xf32>
  %4 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 13} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 12} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 16, 16] -> [1, 3, 4]> : memref<1x3x4xf32> to memref<1x16x16xf32>
  %5 = rock.transform %alloc by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 14} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 12} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 16, 16] -> [1, 2, 4]> : memref<1x2x4xf32> to memref<1x16x16xf32>
  rock.gridwise_gemm_accel(%3, %4, %5) storeMethod( set) features =  mfma|dot|atomic_add {arch = "gfx908:sramecc+:xnack-", blockSize = 64 : i32, gridSize = 1 : i32, numCU = 120 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 16, kpack = 4, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1, forceUnroll = true>} : memref<1x16x16xf32>, memref<1x16x16xf32>, memref<1x16x16xf32>
  %6 = rock.transform %alloc by <affine_map<(d0) -> (0, d0 floordiv 4, d0 mod 4)> by [<Merge{1, 2, 4} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [8] -> [1, 2, 4]> : memref<1x2x4xf32> to memref<8xf32>
  %7 = rock.transform %alloc by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 2} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [2, 4] -> [1, 2, 4]> : memref<1x2x4xf32> to memref<2x4xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : memref<2x4xf32>) outs(%alloc_0 : memref<2x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %10 = arith.mulf %in, %cst : f32
    linalg.yield %10 : f32
  }
  %8 = rock.transform %alloc_0 by <affine_map<(d0, d1, d2) -> (d0 * 2 + d1, d2)> by [<Unmerge{1, 2} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 2, 4] -> [2, 4]> : memref<2x4xf32> to memref<1x2x4xf32>
  %9 = rock.transform %8 by <affine_map<(d0) -> (0, d0 floordiv 4, d0 mod 4)> by [<Merge{1, 2, 4} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [8] -> [1, 2, 4]> : memref<1x2x4xf32> to memref<8xf32>
  memref.copy %6, %arg2 : memref<8xf32> to memref<8xf32>
  memref.copy %9, %arg3 : memref<8xf32> to memref<8xf32>
  return
}
