// RUN: rocmlir-opt -mlir-print-local-scope -split-input-file -rock-gridwise-gemm-to-blockwise -canonicalize -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @gridwise_attn_causal_scale_gqa
func.func @gridwise_attn_causal_scale_gqa(%arg0: memref<8192xf16>, %arg1: memref<8388608xf16>, %arg2: memref<8388608xf16>, %arg3: memref<524288xf16>, %arg4: memref<64xf16>, %arg5: memref<8192xf16>) attributes {block_size = 32 : i32, features = #rock<GemmFeatures wmma|dot|atomic_add|atomic_fmax_f32>, grid_size = 8 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0 * 128 + d2)> by [<Unmerge{64, 128} ["g", "head_qk"] at [0, 2] -> ["raw"] at [0]>, <AddDim{1} ["seq_q"] at [1] -> [] at []>] bounds = [64, 1, 128] -> [8192]> : memref<8192xf16> to memref<64x1x128xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 128 + d1) * 8192 + d2)> by [<Unmerge{8, 128, 8192} ["g", "head_qk", "seq_k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [8, 128, 8192] -> [8388608]> : memref<8388608xf16> to memref<8x128x8192xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 8192 + d1) * 128 + d2)> by [<Unmerge{8, 8192, 128} ["g", "seq_k", "head_v"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [8, 8192, 128] -> [8388608]> : memref<8388608xf16> to memref<8x8192x128xf16>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d0 * 8192 + d2)> by [<Unmerge{64, 8192} ["g", "seq_k"] at [0, 2] -> ["raw"] at [0]>, <AddDim{1} ["seq_q"] at [1] -> [] at []>] bounds = [64, 1, 8192] -> [524288]> : memref<524288xf16> to memref<64x1x8192xf16>
  %4 = rock.transform %arg4 by <affine_map<(d0, d1) -> (d0)> by [<Unmerge{64} ["g"] at [0] -> ["raw"] at [0]>, <AddDim{1} ["seq_q"] at [1] -> [] at []>] bounds = [64, 1] -> [64]> : memref<64xf16> to memref<64x1xf16>
  %5 = rock.transform %arg5 by <affine_map<(d0, d1, d2) -> (d0 * 128 + d2)> by [<Unmerge{64, 128} ["g", "head_v"] at [0, 2] -> ["raw"] at [0]>, <AddDim{1} ["seq_q"] at [1] -> [] at []>] bounds = [64, 1, 128] -> [8192]> : memref<8192xf16> to memref<64x1x128xf16>
  %6 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [64, 128, 1] -> [64, 1, 128]> : memref<64x1x128xf16> to memref<64x128x1xf16>
  %7 = rock.transform %6 by <affine_map<(d0, d1, d2, d3) -> (d0 * 8 + d3, d1, d2)> by [<Unmerge{8, 8} ["gemmG", "numRepeats"] at [0, 3] -> ["gemmG"] at [0]>, <PassThrough ["seqLen", "headDim"] at [2, 1] -> ["seqLen", "headDim"] at [2, 1]>] bounds = [8, 128, 1, 8] -> [64, 128, 1]> : memref<64x128x1xf16> to memref<8x128x1x8xf16>
  %8 = rock.transform %7 by <affine_map<(d0, d1, d2) -> (d0, d1, 0, d2)> by [<Merge{1, 8} ["seqLen"] at [2] -> ["seqLen", "numRepeats"] at [2, 3]>, <PassThrough ["gemmG", "headDim"] at [0, 1] -> ["gemmG", "headDim"] at [0, 1]>] bounds = [8, 128, 8] -> [8, 128, 1, 8]> : memref<8x128x1x8xf16> to memref<8x128x8xf16>
  %9 = rock.transform %5 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 8 + d3 + d1, d2, d4)> by [<Unmerge{8, 8, 1} ["gemmG", "numRepeats", "splitKV"] at [0, 3, 1] -> ["gemmG"] at [0]>, <PassThrough ["seqLen", "headDim"] at [2, 4] -> ["seqLen", "headDim"] at [1, 2]>] bounds = [8, 1, 1, 8, 128] -> [64, 1, 128]> : memref<64x1x128xf16> to memref<8x1x1x8x128xf16>
  %10 = rock.transform %9 by <affine_map<(d0, d1, d2) -> (d0, 0, 0, d1, d2)> by [<Merge{1, 8} ["seqLen"] at [1] -> ["seqLen", "numRepeats"] at [2, 3]>, <Merge{8, 1} ["gemmG"] at [0] -> ["gemmG", "splitKV"] at [0, 1]>, <PassThrough ["headDim"] at [2] -> ["headDim"] at [4]>] bounds = [8, 8, 128] -> [8, 1, 1, 8, 128]> : memref<8x1x1x8x128xf16> to memref<8x8x128xf16>
  %11 = rock.transform %4 by <affine_map<(d0, d1, d2, d3) -> (d0 * 8 + d3 + d1, d2)> by [<Unmerge{8, 8, 1} ["gemmG", "numRepeats", "splitKV"] at [0, 3, 1] -> ["gemmG"] at [0]>, <PassThrough ["seqLen"] at [2] -> ["seqLen"] at [1]>] bounds = [8, 1, 1, 8] -> [64, 1]> : memref<64x1xf16> to memref<8x1x1x8xf16>
  %12 = rock.transform %11 by <affine_map<(d0, d1) -> (d0, 0, 0, d1)> by [<Merge{1, 8} ["seqLen"] at [1] -> ["seqLen", "numRepeats"] at [2, 3]>, <Merge{8, 1} ["gemmG"] at [0] -> ["gemmG", "splitKV"] at [0, 1]>] bounds = [8, 8] -> [8, 1, 1, 8]> : memref<8x1x1x8xf16> to memref<8x8xf16>
  %13 = rock.transform %8 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K"] at [1] -> ["gemm0K"] at [1]>, <Pad{0, 24} ["gemm0NPad"] at [2] -> ["gemm0N"] at [2]>] bounds = [8, 128, 32] -> [8, 128, 8]> : memref<8x128x8xf16> to memref<8x128x32xf16>
  %14 = rock.transform %10 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 24} ["gemm1NPad"] at [1] -> ["gemm1N"] at [1]>, <PassThrough ["gemm1M"] at [2] -> ["gemm1M"] at [2]>] bounds = [8, 32, 128] -> [8, 8, 128]> : memref<8x8x128xf16> to memref<8x32x128xf16>
  %15 = rock.transform %12 by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 24} ["gemm1NPad"] at [1] -> ["gemm1N"] at [1]>] bounds = [8, 32] -> [8, 8]> : memref<8x8xf16> to memref<8x32xf16>

  // CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[c4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[c8:.+]] = arith.constant 8 : index

  // main loop
  // CHECK: scf.for %arg6 = %c0 to %c4 step %c1
  
  // data conversion transforming_for
  // CHECK: rock.transforming_for
  // CHECK: rock.in_bounds_store %{{.*}} -> %[[gemmOut:.+]][{{.*}}]

  // fusion
  // CHECK: %[[loadInto:.+]] = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
  // CHECK: rock.threadwise_read_into
  // CHECK-SAME: #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, 0, d2)> by [<Merge{1, 8} ["seqLenQ"] at [1] -> ["seqLenQ", "numRepeats"] at [2, 1]>, <PassThrough ["gemmG", "seqLenKV"] at [0, 2] -> ["gemmG", "seqLenKV"] at [0, 3]>] bounds = [8, 8, 8192] -> [8, 8, 1, 8192]>, #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d0 * 8 + d1, d2, d3)> by [<Unmerge{8, 8} ["gemmG", "numRepeats"] at [0, 1] -> ["gemmG"] at [0]>, <PassThrough ["seqLenQ", "seqLenKV"] at [2, 3] -> ["seqLenQ", "seqLenKV"] at [1, 2]>] bounds = [8, 8, 1, 8192] -> [64, 1, 8192]>
  // CHECK-SAME: -> %[[loadInto]]
  
  // CHECK: %[[fusionRes:.+]] = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
  // CHECK-NEXT: linalg.generic {{.*}} ins(%[[gemmOut]], %[[loadInto]] : memref<32xf16, #gpu.address_space<private>>, memref<32xf16, #gpu.address_space<private>>) outs(%[[fusionRes]] : memref<32xf16, #gpu.address_space<private>>) attrs =  {rock.majorTensorNumber = 0 : index} {
  // CHECK: %[[fusionOut:.+]] = arith.addf
  // CHECK-NEXT: linalg.yield %[[fusionOut]] : f16


  // padding transforming_for
  // CHECK: rock.transforming_for
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs} (%[[dim0:.+]], %[[dim1:.+]], %[[dim2:.+]]) = [{{.*}}]({{.*}}), ({{.*}}) = []
  // CHECK-NEXT: %[[NIndexDivByNumRepeatsGQA:.+]] = arith.divui %[[dim1]], %[[c8]] : index
  // CHECK-NEXT: %[[causalSecondComparison:.+]] = arith.cmpi ugt, %[[dim2]], %[[NIndexDivByNumRepeatsGQA]] : index
  // CHECK-NEXT: scf.if %[[causalSecondComparison]] {
  // CHECK-NEXT: rock.in_bounds_store
  rock.gridwise_attention_accel(%13, %1, %2, %3, %14, %15) features =  wmma|dot|atomic_add|atomic_fmax_f32 preSoftmaxOps = {
  ^bb0(%arg6: memref<64x1x8192xf16>, %arg7: memref<64x1x8192xf16>, %arg8: memref<64x1x8192xf16>):
    %16 = rock.transform %arg6 by <affine_map<(d0, d1) -> (d0, 0, d1)> by [<Merge{64, 1} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [64, 8192] -> [64, 1, 8192]> : memref<64x1x8192xf16> to memref<64x8192xf16>
    %17 = rock.transform %arg7 by <affine_map<(d0, d1) -> (d0, 0, d1)> by [<Merge{64, 1} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [64, 8192] -> [64, 1, 8192]> : memref<64x1x8192xf16> to memref<64x8192xf16>
    %alloc = memref.alloc() : memref<64x1x8192xf16>
    %18 = rock.transform %alloc by <affine_map<(d0, d1) -> (d0, 0, d1)> by [<Merge{64} ["dim0"] at [0] -> ["exp0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <ConstDim{0, 1} [] at [] -> ["unit1"] at [1]>] bounds = [64, 8192] -> [64, 1, 8192]> : memref<64x1x8192xf16> to memref<64x8192xf16>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%16, %17 : memref<64x8192xf16>, memref<64x8192xf16>) outs(%18 : memref<64x8192xf16>) attrs =  {rock.majorTensorNumber = 0 : index} {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %19 = arith.addf %in, %in_0 : f16
      linalg.yield %19 : f16
    }
    memref.copy %alloc, %arg8 : memref<64x1x8192xf16> to memref<64x1x8192xf16>
    rock.yield
  } {blockSize = 32 : i32, causal, firstGemmIndices = array<i64: 0>, gridSize = 8 : i32, numRepeatsGQA = 8 : index, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1>, params0 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll  = true>, params1 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll  = true>, prePadG0N = 8 : index, softmaxType = f32, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>} : memref<8x128x32xf16>, memref<8x128x8192xf16>, memref<8x8192x128xf16>, memref<64x1x8192xf16>, memref<8x32x128xf16>, memref<8x32xf16>
  return
}
