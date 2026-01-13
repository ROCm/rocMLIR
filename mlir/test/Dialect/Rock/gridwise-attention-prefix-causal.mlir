// RUN: rocmlir-opt --rock-gridwise-gemm-to-blockwise %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4) -> ((d1 * 16 + d3) * 64 + d4)>
#map1 = affine_map<(d0, d1, d2, d3) -> ((d1 * 18 + d2) * 64 + d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>
#map6 = affine_map<(d0, d1, d2) -> (0, d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2) -> (0, d0 floordiv 7, d0 mod 7, d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> ((d0 * 16 + d1) * 64 + d2)>
#map9 = affine_map<(d0, d1, d2, d3) -> (((d0 * 4 + d1) * 14 + d2) * 64 + d3)>
#map10 = affine_map<(d0, d1) -> (d1)>
#map11 = affine_map<(d0, d1) -> (d0, 0)>
#map12 = affine_map<(d0) -> (0, d0)>
#map13 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map14 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map15 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map16 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#accel_gemm_params = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>
#transform_map = #rock.transform_map<#map by [<Unmerge{2, 16, 64} ["exp1", "exp3", "exp4"] at [1, 3, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>] bounds = [1, 2, 1, 16, 64] -> [2048]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{4, 18, 64} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 4, 18, 64] -> [4608]>
#transform_map2 = #rock.transform_map<#map2 by [<PassThrough ["dim0", "dim2", "dim1", "dim3"] at [0, 1, 2, 3] -> ["dim0", "dim2", "dim1", "dim3"] at [0, 2, 1, 3]>] bounds = [1, 18, 4, 64] -> [1, 4, 18, 64]>
#transform_map3 = #rock.transform_map<#map3 by [<Slice{0, 1, 0, 14, 0, 4, 0, 64} ["dim0_sliced", "dim1_sliced", "dim2_sliced", "dim3_sliced"] at [0, 1, 2, 3] -> ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3]>] bounds = [1, 14, 4, 64] -> [1, 18, 4, 64]>
#transform_map4 = #rock.transform_map<#map4 by [<PassThrough ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 4, 3]>] bounds = [1, 2, 1, 64, 16] -> [1, 2, 1, 16, 64]>
#transform_map5 = #rock.transform_map<#map5 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [1, 2, 7, 64, 16] -> [1, 2, 1, 64, 16]>
#transform_map6 = #rock.transform_map<#map6 by [<Merge{1, 14} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [14, 4, 64] -> [1, 14, 4, 64]>
#transform_map7 = #rock.transform_map<#map7 by [<Merge{1, 2, 7} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [14, 64, 16] -> [1, 2, 7, 64, 16]>
#transform_map8 = #rock.transform_map<#map8 by [<Unmerge{14, 16, 64} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [14, 16, 64] -> [14336]>
#transform_map9 = #rock.transform_map<#map9 by [<Unmerge{1, 4, 14, 64} ["col0", "col1", "col2", "col3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [1, 4, 14, 64] -> [3584]>
#transform_map10 = #rock.transform_map<#map2 by [<PassThrough ["dim0", "dim2", "dim1", "dim3"] at [0, 2, 1, 3] -> ["dim0", "dim2", "dim1", "dim3"] at [0, 1, 2, 3]>] bounds = [1, 14, 4, 64] -> [1, 4, 14, 64]>
#transform_map11 = #rock.transform_map<#map6 by [<Merge{14} ["dim0"] at [0] -> ["exp1"] at [1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>, <ConstDim{0, 1} [] at [] -> ["unit0"] at [0]>] bounds = [14, 4, 64] -> [1, 14, 4, 64]>
#transform_map12 = #rock.transform_map<#map10 by [<Unmerge{1} ["exp1"] at [1] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 1] -> [1]>
#transform_map13 = #rock.transform_map<#map11 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>] bounds = [1, 14] -> [1, 1]>
#transform_map14 = #rock.transform_map<#map12 by [<Merge{1, 14} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>] bounds = [14] -> [1, 14]>
#transform_map15 = #rock.transform_map<#map13 by [<PassThrough ["dim1", "dim0", "dim2"] at [0, 1, 2] -> ["dim1", "dim0", "dim2"] at [1, 0, 2]>] bounds = [4, 14, 64] -> [14, 4, 64]>
#transform_map16 = #rock.transform_map<#map13 by [<PassThrough ["dim1", "dim0", "dim2"] at [0, 1, 2] -> ["dim1", "dim0", "dim2"] at [1, 0, 2]>] bounds = [14, 4, 64] -> [4, 14, 64]>
#transform_map17 = #rock.transform_map<#map14 by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [14, 16, 64] -> [14, 64, 16]>
#transform_map18 = #rock.transform_map<#map14 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [14, 64, 4] -> [14, 4, 64]>
#transform_map19 = #rock.transform_map<#map14 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0N"] at [1, 2] -> ["gemm0K", "gemm0N"] at [2, 1]>] bounds = [14, 64, 16] -> [14, 16, 64]>
#transform_map20 = #rock.transform_map<#map15 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K"] at [1] -> ["gemm0K"] at [1]>, <Pad{0, 28} ["gemm0NPad"] at [2] -> ["gemm0N"] at [2]>] bounds = [14, 64, 32] -> [14, 64, 4]>
#transform_map21 = #rock.transform_map<#map15 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K"] at [1] -> ["gemm0K"] at [1]>, <Pad{0, 16} ["gemm0MPad"] at [2] -> ["gemm0M"] at [2]>] bounds = [14, 64, 32] -> [14, 64, 16]>
#transform_map22 = #rock.transform_map<#map15 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 16} ["gemm1KPad"] at [1] -> ["gemm1K"] at [1]>, <PassThrough ["gemm1M"] at [2] -> ["gemm1M"] at [2]>] bounds = [14, 32, 64] -> [14, 16, 64]>
#transform_map23 = #rock.transform_map<#map15 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 28} ["gemm1NPad"] at [1] -> ["gemm1N"] at [1]>, <PassThrough ["gemm1M"] at [2] -> ["gemm1M"] at [2]>] bounds = [14, 32, 64] -> [14, 4, 64]>
#transform_map24 = #rock.transform_map<#map16 by [<Unmerge{14} ["exp1"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 14, 4, 16] -> [14, 4, 16]>
#transform_map25 = #rock.transform_map<#map6 by [<Merge{1, 14} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [14, 4, 16] -> [1, 14, 4, 16]>
#transform_map26 = #rock.transform_map<#map6 by [<Merge{14} ["dim0"] at [0] -> ["exp1"] at [1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>, <ConstDim{0, 1} [] at [] -> ["unit0"] at [0]>] bounds = [14, 4, 16] -> [1, 14, 4, 16]>
module {
  // CHECK-LABEL: func @mlir_attention
  // Verify prefix causal loop bound calculation: effectiveSeqLen = min(maxRowOfBlock + prefixOffset, gemm0M - 1)
  // The muli computes n_block * blockSize, then subi computes maxRowOfBlock = nextBlockStart - 1
  // CHECK: arith.muli %{{.*}}, %c32{{.*}} : index
  // CHECK: %[[MAX_ROW:.*]] = arith.subi %{{.*}}, %c1{{.*}} : index
  // The prefixOffset is loaded from tensor and converted to index
  // CHECK: %[[OFFSET:.*]] = arith.index_cast %{{.*}} : i32 to index
  // effectiveSeqLen = maxRowOfBlock + prefixOffset
  // CHECK: %[[EFFECTIVE_SEQ_UNBOUND:.*]] = arith.addi %[[MAX_ROW]], %[[OFFSET]] : index
  // Bound by gemm0M - 1 (key sequence length - 1) to prevent out-of-bounds access
  // CHECK: %[[EFFECTIVE_SEQ:.*]] = arith.minui %[[EFFECTIVE_SEQ_UNBOUND]], %c{{.*}} : index
  // Ceiling division: (effectiveSeqLen + blockSize) / blockSize
  // CHECK: arith.addi %[[EFFECTIVE_SEQ]], %c32{{.*}} : index
  // CHECK: %[[END:.*]] = arith.divui %{{.*}}, %c32{{.*}} : index

  // Verify the main loop uses the computed end bound
  // CHECK: scf.for %{{.*}} = %c0{{.*}} to %[[END]] step %c1{{.*}}

  // Verify prefix causal masking: key_pos > query_pos + prefixOffset
  // CHECK: %[[COL_PLUS_OFFSET:.*]] = arith.addi %{{.*}}, %[[OFFSET]] : index
  // CHECK: %[[MASK_COND:.*]] = arith.cmpi ugt, %{{.*}}, %[[COL_PLUS_OFFSET]] : index
  // CHECK: scf.if %[[MASK_COND]]
  // CHECK:   rock.in_bounds_store %{{.*}} -> %{{.*}}

  func.func @mlir_attention(%arg0: memref<1xi32>, %arg1: memref<4608xf16>, %arg2: memref<2048xf16>, %arg3: memref<14336xf16>, %arg4: memref<3584xf16>) attributes {arch = "gfx950", block_size = 64 : i32, grid_size = 14 : i32, kernel} {
    %cst = arith.constant 1.250000e-01 : f16
    %0 = rock.transform %arg2 by #transform_map : memref<2048xf16> to memref<1x2x1x16x64xf16>
    %1 = rock.transform %arg1 by #transform_map1 : memref<4608xf16> to memref<1x4x18x64xf16>
    %2 = rock.transform %1 by #transform_map2 : memref<1x4x18x64xf16> to memref<1x18x4x64xf16>
    %3 = rock.transform %2 by #transform_map3 : memref<1x18x4x64xf16> to memref<1x14x4x64xf16>
    %4 = rock.transform %0 by #transform_map4 : memref<1x2x1x16x64xf16> to memref<1x2x1x64x16xf16>
    %5 = rock.transform %4 by #transform_map5 : memref<1x2x1x64x16xf16> to memref<1x2x7x64x16xf16>
    %6 = rock.transform %3 by #transform_map6 : memref<1x14x4x64xf16> to memref<14x4x64xf16>
    %7 = rock.transform %5 by #transform_map7 : memref<1x2x7x64x16xf16> to memref<14x64x16xf16>
    %8 = rock.transform %arg3 by #transform_map8 : memref<14336xf16> to memref<14x16x64xf16>
    %alloc = memref.alloc() : memref<3584xf16>
    %9 = rock.transform %alloc by #transform_map9 : memref<3584xf16> to memref<1x4x14x64xf16>
    %10 = rock.transform %9 by #transform_map10 : memref<1x4x14x64xf16> to memref<1x14x4x64xf16>
    %11 = rock.transform %10 by #transform_map11 : memref<1x14x4x64xf16> to memref<14x4x64xf16>
    %12 = rock.transform %arg0 by #transform_map12 : memref<1xi32> to memref<1x1xi32>
    %13 = rock.transform %12 by #transform_map13 : memref<1x1xi32> to memref<1x14xi32>
    %14 = rock.transform %13 by #transform_map14 : memref<1x14xi32> to memref<14xi32>
    %15 = rock.transform %6 by #transform_map15 : memref<14x4x64xf16> to memref<4x14x64xf16>
    %16 = rock.transform %15 by #transform_map16 : memref<4x14x64xf16> to memref<14x4x64xf16>
    %17 = rock.transform %7 by #transform_map17 : memref<14x64x16xf16> to memref<14x16x64xf16>
    %18 = rock.transform %16 by #transform_map18 : memref<14x4x64xf16> to memref<14x64x4xf16>
    %19 = rock.transform %17 by #transform_map19 : memref<14x16x64xf16> to memref<14x64x16xf16>
    %20 = rock.transform %18 by #transform_map20 : memref<14x64x4xf16> to memref<14x64x32xf16>
    %21 = rock.transform %19 by #transform_map21 : memref<14x64x16xf16> to memref<14x64x32xf16>
    %22 = rock.transform %8 by #transform_map22 : memref<14x16x64xf16> to memref<14x32x64xf16>
    %23 = rock.transform %11 by #transform_map23 : memref<14x4x64xf16> to memref<14x32x64xf16>
    rock.gridwise_attention_accel(%20, %21, %22, %14, %23) preSoftmaxOps = {
    ^bb0(%arg5: memref<14x4x16xf16>, %arg6: memref<1x14x4x16xf16>):
      %24 = rock.transform %arg5 by #transform_map24 : memref<14x4x16xf16> to memref<1x14x4x16xf16>
      %25 = rock.transform %24 by #transform_map25 : memref<1x14x4x16xf16> to memref<14x4x16xf16>
      %alloc_0 = memref.alloc() : memref<1x14x4x16xf16>
      %26 = rock.transform %alloc_0 by #transform_map26 : memref<1x14x4x16xf16> to memref<14x4x16xf16>
      linalg.generic {indexing_maps = [#map15, #map15], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25 : memref<14x4x16xf16>) outs(%26 : memref<14x4x16xf16>) attrs =  {rock.majorTensorNumber = 0 : index} {
      ^bb0(%in: f16, %out: f16):
        %27 = arith.mulf %in, %cst : f16
        linalg.yield %27 : f16
      }
      memref.copy %alloc_0, %arg6 : memref<1x14x4x16xf16> to memref<1x14x4x16xf16>
      rock.yield
    } {blockSize = 64 : i32, causal, firstGemmIndices = array<i64: 0>, gridSize = 14 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1, 1, 0>, params0 = #accel_gemm_params, params1 = #accel_gemm_params, prePadG0M = 16 : index, prePadG0N = 4 : index, softmaxType = f32, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>} : memref<14x64x32xf16>, memref<14x64x32xf16>, memref<14x32x64xf16>, memref<14xi32>, memref<14x32x64xf16>
    memref.copy %alloc, %arg4 : memref<3584xf16> to memref<3584xf16>
    return
  }
}

