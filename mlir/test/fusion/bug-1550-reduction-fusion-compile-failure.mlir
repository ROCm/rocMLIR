// RUN: rocmlir-opt --rock-regularize --mlir-print-local-scope %s | FileCheck %s
// RUN: rocmlir-driver --kernel-pipeline=gpu --arch "gfx908:sramecc+:xnack-"

// -----// IR Dump After RockGemmToGridwisePass (rock-gemm-to-gridwise) //----- //
// CHECK-LABEL: @mlir_convolution_reshape_mul_reshape_reduce_sum_reshape_mul_mul_reshape_reduce_sum_reshape
func.func @mlir_convolution_reshape_mul_reshape_reduce_sum_reshape_mul_mul_reshape_reduce_sum_reshape(%arg0: memref<320xf32>, %arg1: memref<32768xf32>, %arg2: memref<11520xf32>, %arg3: memref<64xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32}, %arg4: memref<64xf32>, %arg5: memref<2621440xf32>) attributes {arch = "gfx908:sramecc+:xnack-", block_size = 256 : i32, grid_size = 1536 : i32, kernel = "mixr", num_cu = 120 : i64} {
  %cst = arith.constant 2.44140629E-5 : f32
  %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 4 + d1) * 64 + d2) * 64 + d3)> by [<Unmerge{2, 4, 64, 64} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [2, 4, 64, 64] -> [32768]> : memref<32768xf32> to memref<2x4x64x64xf32>
  %1 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 4 + d1) * 3 + d2) * 3 + d3)> by [<Unmerge{320, 4, 3, 3} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [320, 4, 3, 3] -> [11520]> : memref<11520xf32> to memref<320x4x3x3xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x320x64x64xf32>
  %2 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 4 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 4} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [2, 1, 4, 64, 64] -> [2, 4, 64, 64]> : memref<2x4x64x64xf32> to memref<2x1x4x64x64xf32>
  %3 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 320 + d1, d2, d3, d4)> by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 320} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 320, 4, 3, 3] -> [320, 4, 3, 3]> : memref<320x4x3x3xf32> to memref<1x320x4x3x3xf32>
  %4 = rock.transform %alloc by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 320 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 320} ["g", "k"] at [1, 2] -> ["k"] at [1]>] bounds = [2, 1, 320, 64, 64] -> [2, 320, 64, 64]> : memref<2x320x64x64xf32> to memref<2x1x320x64x64xf32>
  %5 = rock.transform %3 by <affine_map<(d0, d1, d2) -> (d0, d2, d1 floordiv 9, (d1 mod 9) floordiv 3, d1 mod 3)> by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <Merge{4, 3, 3} ["gemmK"] at [1] -> ["c", "0", "1"] at [2, 3, 4]>, <PassThrough ["gemmM"] at [2] -> ["k"] at [1]>] bounds = [1, 36, 320] -> [1, 320, 4, 3, 3]> : memref<1x320x4x3x3xf32> to memref<1x36x320xf32>
  %6 = rock.transform %2 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 - 1, d4 - 1)> by [<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, <PassThrough ["gi"] at [1] -> ["gi"] at [1]>, <PassThrough ["ci"] at [2] -> ["ci"] at [2]>, <Pad{1, 1, 1, 1} ["0ipad", "1ipad"] at [3, 4] -> ["0i", "1i"] at [3, 4]>] bounds = [2, 1, 4, 66, 66] -> [2, 1, 4, 64, 64]> : memref<2x1x4x64x64xf32> to memref<2x1x4x66x66xf32>
  %7 = rock.transform %6 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4, d5 + d6)> by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Embed{1, 1} ["0", "0o"] at [3, 4] -> ["0ipad"] at [3]>, <Embed{1, 1} ["1", "1o"] at [5, 6] -> ["1ipad"] at [4]>] bounds = [2, 1, 4, 3, 64, 3, 64] -> [2, 1, 4, 66, 66]> : memref<2x1x4x66x66xf32> to memref<2x1x4x3x64x3x64xf32>
  %8 = rock.transform %7 by <affine_map<(d0, d1, d2) -> (d2 floordiv 4096, d0, d1 floordiv 9, (d1 mod 9) floordiv 3, (d2 mod 4096) floordiv 64, d1 mod 3, d2 mod 64)> by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{4, 3, 3} ["gemmK"] at [1] -> ["ci", "0", "1"] at [2, 3, 5]>, <Merge{2, 64, 64} ["gemmN"] at [2] -> ["ni", "0o", "1o"] at [0, 4, 6]>] bounds = [1, 36, 8192] -> [2, 1, 4, 3, 64, 3, 64]> : memref<2x1x4x3x64x3x64xf32> to memref<1x36x8192xf32>
  %9 = rock.transform %4 by <affine_map<(d0, d1, d2) -> (d2 floordiv 4096, d0, d1, (d2 mod 4096) floordiv 64, d2 mod 64)> by [<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, <PassThrough ["gemmM"] at [1] -> ["ko"] at [2]>, <Merge{2, 64, 64} ["gemmN"] at [2] -> ["no", "0o", "1o"] at [0, 3, 4]>] bounds = [1, 320, 8192] -> [2, 1, 320, 64, 64]> : memref<2x1x320x64x64xf32> to memref<1x320x8192xf32>
  %10 = rock.transform %5 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 28} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 64} ["gemmMPad"] at [2] -> ["gemmM"] at [2]>] bounds = [1, 64, 384] -> [1, 36, 320]> : memref<1x36x320xf32> to memref<1x64x384xf32>
  %11 = rock.transform %8 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 28} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 64, 8192] -> [1, 36, 8192]> : memref<1x36x8192xf32> to memref<1x64x8192xf32>
  %12 = rock.transform %9 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 64} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 384, 8192] -> [1, 320, 8192]> : memref<1x320x8192xf32> to memref<1x384x8192xf32>
  rock.gridwise_gemm_accel(%10, %11, %12) storeMethod( set) features =  mfma|dot|atomic_add {arch = "gfx908:sramecc+:xnack-", blockSize = 256 : i32, gridSize = 1536 : i32, numCU = 120 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 128, nPerBlock = 16, kpack = 8, mPerWave = 32, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1, forceUnroll = true>} : memref<1x64x384xf32>, memref<1x64x8192xf32>, memref<1x384x8192xf32>
  %13 = rock.transform %alloc by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 10 + d2, d3, d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Unmerge{32, 10} ["exp1", "exp2"] at [1, 2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [4] -> ["dim3"] at [3]>] bounds = [2, 32, 10, 64, 64] -> [2, 320, 64, 64]> : memref<2x320x64x64xf32> to memref<2x32x10x64x64xf32>
  %14 = rock.transform %alloc by <affine_map<(d0) -> (d0 floordiv 1310720, (d0 mod 1310720) floordiv 4096, (d0 mod 4096) floordiv 64, d0 mod 64)> by [<Merge{2, 320, 64, 64} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3"] at [0, 1, 2, 3]>] bounds = [2621440] -> [2, 320, 64, 64]> : memref<2x320x64x64xf32> to memref<2621440xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x32x10x64x64xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%13 : memref<2x32x10x64x64xf32>) outs(%alloc_0 : memref<2x32x10x64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %19 = arith.mulf %in, %cst : f32
    linalg.yield %19 : f32
  }
  %15 = rock.transform %alloc_0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 4096, (d2 mod 4096) floordiv 64, d2 mod 64)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Merge{10, 64, 64} ["dim2"] at [2] -> ["col2", "col3", "col4"] at [2, 3, 4]>] bounds = [2, 32, 40960] -> [2, 32, 10, 64, 64]> : memref<2x32x10x64x64xf32> to memref<2x32x40960xf32>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x32x1xf32>
  rock.reduce  sum %15 into %alloc_1 features =  mfma|dot|atomic_add {axis = 2 : index, blockSize = 256 : i32, gridSize = 2400 : i32} : memref<2x32x40960xf32> into memref<2x32x1xf32>
  %16 = rock.transform %alloc_1 by <affine_map<(d0) -> (d0 floordiv 32, d0 mod 32, 0)> by [<Merge{2, 32, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [64] -> [2, 32, 1]> : memref<2x32x1xf32> to memref<64xf32>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x32x10x64x64xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%13 : memref<2x32x10x64x64xf32>) outs(%alloc_2 : memref<2x32x10x64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %19 = arith.mulf %in, %in : f32
    %20 = arith.mulf %19, %cst : f32
    linalg.yield %20 : f32
  }
  %17 = rock.transform %alloc_2 by <affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 4096, (d2 mod 4096) floordiv 64, d2 mod 64)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Merge{10, 64, 64} ["dim2"] at [2] -> ["col2", "col3", "col4"] at [2, 3, 4]>] bounds = [2, 32, 40960] -> [2, 32, 10, 64, 64]> : memref<2x32x10x64x64xf32> to memref<2x32x40960xf32>
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<2x32x1xf32>
  rock.reduce  sum %17 into %alloc_3 features =  mfma|dot|atomic_add {axis = 2 : index, blockSize = 256 : i32, gridSize = 2400 : i32} : memref<2x32x40960xf32> into memref<2x32x1xf32>
  %18 = rock.transform %alloc_3 by <affine_map<(d0) -> (d0 floordiv 32, d0 mod 32, 0)> by [<Merge{2, 32, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [64] -> [2, 32, 1]> : memref<2x32x1xf32> to memref<64xf32>
  memref.copy %16, %arg3 : memref<64xf32> to memref<64xf32>
  memref.copy %18, %arg4 : memref<64xf32> to memref<64xf32>
  memref.copy %14, %arg5 : memref<2621440xf32> to memref<2621440xf32>
  return
}

