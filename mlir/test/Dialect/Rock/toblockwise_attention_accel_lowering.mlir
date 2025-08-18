// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -canonicalize -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @gridwise_attn_simple
// CHECK-SAME: (%[[Q:.+]]: memref<1x384x64xf32>, %[[K:.+]]: memref<1x64x384xf32>, %[[V:.+]]: memref<1x384x64xf32>, %[[O:.+]]: memref<1x384x64xf32>)
// CHECK-DAG: %[[ln2Recip:.+]] = arith.constant 1.44269502 : f32
// CHECK-DAG: %[[negInf:.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG: %[[zeroF32:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[zeroVecF32:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>

// CHECK-DAG: %[[QTr0:.+]] = rock.transform %[[Q]] by

// init maxRow buffer
// CHECK-DAG: rock.fill(%[[maxRowBuf:.+]], %[[negInf]])

// init sumRow buffer
// CHECK-DAG: rock.fill(%[[sumRowBuf:.+]], %[[zeroF32]])

// init attentionAcc buffer
// CHECK-DAG: rock.fill(%[[attnOutBuf:.+]], %[[zeroF32]])

// Outer N-tile loop
// CHECK: affine.for
  // CHECK-DAG: rock.fill(%[[gemm0AccBuf:.+]], %[[zeroVecF32]])
  // CHECK: %[[ldsG0A:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ldsG0B:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  
  // Inner gemm0 KpacksPerBlock loop
  // CHECK: affine.for
    // CHECK: rock.lds_barrier
    // CHECK: rock.blockwise_load_tile %[[QTr0]]{{.*}} LDS -> %[[ldsG0A]] -> %[[preAccelRegA:[0-9]+]] {{.*}}#rock<GemmLoadTileType DoubleBuffer>

    // CHECK-DAG: %[[viewG0AStore:.+]] = memref.view %[[ldsG0A]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    
    // CHECK: rock.blockwise_load_tile %[[K]]{{.*}} LDS -> %[[ldsG0B]] -> %[[preAccelRegB:[0-9]+]] {{.*}}#rock<GemmLoadTileType Default>

    // CHECK: %[[viewG0BStore:.+]] = memref.view %[[ldsG0B]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    // CHECK: rock.lds_barrier

    // Emit blockwise gemm0
    // CHECK: rock.blockwise_gemm_accel %[[gemm0AccBuf]] += %[[preAccelRegB]] from %[[viewG0BStore]] * %[[preAccelRegA]] from %[[viewG0AStore]]
    // CHECK-SAME: loadAfromLDS
    // CHECK-NOT: loadBfromLDS

  // CHECK: rock.transforming_for
    // CHECK: %[[tmp:.+]] =  memref.load %[[gemm0AccBuf]][
    // CHECK: rock.in_bounds_store %[[tmp]] -> %[[gemm0AccBufScalar:.+]][
  // CHECK: linalg.generic {{.*}} ins(%[[gemm0AccBufScalar]] {{.*}} outs(%[[gemm0AccBufScalar]]
    // CHECK: %[[gemm0Scaled:.+]] = arith.mulf %in, %[[ln2Recip]] : f32
    // CHECK: linalg.yield %[[gemm0Scaled]]
  // CHECK: %[[ldsReductionWS:.+]] = rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ldsReductionWSView:.+]] = memref.view %[[ldsReductionWS]][{{.*}}][] : memref<256xi8, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
  // CHECK: rock.blockwise_broadcast_reduce max {{.*}} %[[gemm0AccBufScalar]] into %[[gemm0Max:[0-9]+]] using %[[ldsReductionWSView]]

  // Compute exp(gemm0 - rowmax_j)
  // *****************************
  // CHECK: rock.transforming_for
    // CHECK-DAG: %[[rowmax:.+]] = rock.in_bounds_load %[[maxRowBuf]]
    // CHECK-DAG: %[[tilemax:.+]] = rock.in_bounds_load %[[gemm0Max]]
    // CHECK-DAG: %[[newmax:.+]] = arith.maximumf %[[rowmax]], %[[tilemax]]
    // CHECK-DAG: %[[gemm0Val:.+]] = rock.in_bounds_load %[[gemm0AccBufScalar]]
    // CHECK-DAG: %[[gemm0ValSubMax:.+]] = arith.subf %[[gemm0Val]], %[[newmax]]
    // CHECK-DAG: %[[gemm0ValSubMaxExp:.+]] = math.exp2 %[[gemm0ValSubMax]]
    // CHECK-DAG: rock.in_bounds_store %[[gemm0ValSubMaxExp]] -> %[[gemm0NormExp:.+]][

  // CHECK: %[[ldsReductionWS2:.+]] = rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ldsReductionWS2View:.+]] = memref.view %[[ldsReductionWS2]][{{.*}}][] : memref<256xi8, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
  // CHECK: rock.blockwise_broadcast_reduce sum {{.*}} %[[gemm0NormExp]] into %[[gemm0NormExpSum:[0-9]+]] using %[[ldsReductionWS2View]]

  // li = exp(m_{j-1} - m_{j}) * l_{j-1} + rowsum(Pij)
  // where
  // l is the rowsum accumulator
  // m is the rowmax accmulator
  // P is exp(gemm0 - rowmax_j)
  // *************************************************
  // CHECK: rock.transforming_for
    // CHECK-DAG: %[[rowsum:.+]] = rock.in_bounds_load %[[sumRowBuf]]
    // CHECK-DAG: %[[tilesum:.+]] = rock.in_bounds_load %[[gemm0NormExpSum]]
    // CHECK-DAG: %[[rowmax:.+]] = rock.in_bounds_load %[[maxRowBuf]]
    // CHECK-DAG: %[[tilemax:.+]] = rock.in_bounds_load %[[gemm0Max]]
    // CHECK-DAG: %[[newmax:.+]] = arith.maximumf %[[rowmax]], %[[tilemax]]
    // CHECK-DAG: %[[maxdiff:.+]] = arith.subf %[[rowmax]], %[[newmax]]
    // CHECK-DAG: %[[maxdiffexp:.+]] =  math.exp2 %[[maxdiff]]
    // CHECK-DAG: rock.in_bounds_store %[[maxdiffexp]] -> %[[maxdiffexpbuf:.+]][
    // CHECK-DAG: %[[rowsummul:.+]] =  arith.mulf %[[maxdiffexp]], %[[rowsum]]
    // CHECK-DAG: %[[tilesumadd:.+]] =  arith.addf %[[rowsummul]], %[[tilesum]]
    // CHECK-DAG: %[[tilesumadd]] -> %[[sumRowBuf]]

  // Viewing first gemm output as K x D
  // CHECK-DAG: %[[gemm0NormExpTr0:.+]] = rock.transform %[[gemm0NormExp]]
  // CHECK-DAG: %[[gemm0NormExpTr1:.+]] = rock.transform %[[gemm0NormExpTr0]]
  // CHECK-DAG: %[[gemm0NormExpTr2:.+]] = rock.transform %[[gemm0NormExpTr1]]
  // CHECK-DAG: %[[gemm0NormExpTr3:.+]] = rock.transform %[[gemm0NormExpTr2]]
  // CHECK-DAG: %[[gemm0NormExpTr4:.+]] = rock.transform %[[gemm0NormExpTr3]]
  // CHECK-DAG: %[[gemm0NormExpTr5:.+]] = rock.transform %[[gemm0NormExpTr4]]
  
  // CHECK-DAG: %[[ldsG1AStore:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>

  // Viewing another set of register with kPack packing
  // CHECK: %[[G1AregsKpackTr0:.+]] = rock.transform %[[G1AregsKpack:.+]] by
  // CHECK-DAG: %[[G1AregsKpackTr1:.+]] = rock.transform %[[G1AregsKpackTr0]] by
  // CHECK-DAG: %[[G1AregsKpackTr2:.+]] = rock.transform %[[G1AregsKpackTr1]] by
  // CHECK-DAG: %[[G1AregsKpackTr3:.+]] = rock.transform %[[G1AregsKpackTr2]] by
  // CHECK-DAG: %[[G1AregsKpackTr4:.+]] = rock.transform %[[G1AregsKpackTr3]] by
  // CHECK-DAG: %[[G1AregsKpackTr5:.+]] = rock.transform %[[G1AregsKpackTr4]] by

  // CHECK-DAG: rock.threadwise_copy %[[gemm0NormExpTr5]] -> %[[G1AregsKpackTr5]]

  // Viewing G1 LDS A tile buffer
  // CHECK-DAG: %[[viewG1AStore:.+]] = memref.view %[[ldsG1AStore]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[viewG1AStoreTr0:.+]] = rock.transform %[[viewG1AStore]]
  // CHECK-DAG: %[[viewG1AStoreTr1:.+]] = rock.transform %[[viewG1AStoreTr0]]
  // CHECK-DAG: %[[viewG1AStoreTr2:.+]] = rock.transform %[[viewG1AStoreTr1]]
  // CHECK-DAG: %[[viewG1AStoreTr3:.+]] = rock.transform %[[viewG1AStoreTr2]]
  // CHECK-DAG: %[[viewG1AStoreTr4:.+]] = rock.transform %[[viewG1AStoreTr3]]
  // CHECK-DAG: %[[viewG1AStoreTr5:.+]] = rock.transform %[[viewG1AStoreTr4]]
  // CHECK-DAG: %[[viewG1AStoreTr6:.+]] = rock.transform %[[viewG1AStoreTr5]]
  // CHECK-DAG: %[[viewG1AStoreTr7:.+]] = rock.transform %[[viewG1AStoreTr6]]

  // Store to LDS G1A tile buffer
  // CHECK-DAG: rock.threadwise_write_all {{.*}} %[[G1AregsKpack]] -> [](%[[viewG1AStoreTr7]])
  // CHECK-DAG: %[[view2G1AStore:.+]] = memref.view %[[ldsG1AStore]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
  
  // CHECK-DAG: %[[ldsG0BStore:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>

  // Gemm1
  // CHECK: affine.for %[[g1MIter:.+]]
    // CHECK-DAG: rock.fill(%[[gemm1AccBuf:.+]], %[[zeroVecF32]])
    // CHECK: rock.lds_barrier

    // CHECK: rock.blockwise_load_tile %[[V]]{{.*}} LDS -> %[[ldsG0BStore]] -> %[[preAccelRegV:[0-9]+]] {{.*}}#rock<GemmLoadTileType Default>

    // CHECK: %[[view2G1BStore:.+]] = memref.view %[[ldsG0BStore]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>

    // CHECK: rock.lds_barrier

    // Emit blockwise gemm1
    // CHECK: rock.blockwise_gemm_accel %[[gemm1AccBuf]] += %[[preAccelRegV]] from %[[view2G1BStore]] * %[[preAccelRegA:[0-9]+]] from %[[view2G1AStore]]
    // CHECK-SAME: loadAfromLDS
    // CHECK-SAME: loadBfromLDS

    // CHECK: rock.transforming_for
      // CHECK: %[[tmp1:.+]] =  memref.load %[[gemm1AccBuf]][
      // CHECK: rock.in_bounds_store %[[tmp1]] -> %[[gemm1AccBufScalar:.+]][

    // CHECK: %[[sliceAttnOutBuf:.+]] = memref.subview %[[attnOutBuf]]
    // Reduction corrections
    // CHECK: rock.transforming_for
      // CHECK-DAG: %[[maxdiffexp:.+]] = rock.in_bounds_load %[[maxdiffexpbuf]]
      // CHECK-DAG: %[[attnOutVal:.+]] = rock.in_bounds_load %[[sliceAttnOutBuf]]
      // CHECK-DAG: %[[gemm1Val:.+]] = rock.in_bounds_load %[[gemm1AccBufScalar]]

      // CHECK-DAG: %[[attnOutBufMul:.+]] = arith.mulf %[[attnOutVal]], %[[maxdiffexp]]
      // CHECK-DAG: %[[newattnOutVal:.+]] = arith.addf %[[attnOutBufMul]], %[[gemm1Val]]
      // CHECK-DAG: rock.in_bounds_store %[[newattnOutVal]] -> %[[sliceAttnOutBuf]]
    // CHECK : }
  // CHECK : }
// CHECK : }
// CHECK : %[[flatAttnOutBuf:.+]] = memref.collapse_shape %[[attnOutBuf]]
// CHECK : rock.threadwise_write_all {{.*}} %[[flatAttnOutBuf]] -> {{.*}}(%[[O]])

func.func @gridwise_attn_simple(%arg0: memref<1x384x64xf32>, %arg1: memref<1x64x384xf32>, %arg2: memref<1x384x64xf32>, %arg3: memref<1x384x64xf32>) attributes {block_size = 64 : i32, grid_size = 24 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>
  rock.gridwise_attention_accel(%0, %arg1, %arg2, %arg3) preSoftmaxOps = {} {
    blockSize = 64 : i32,
    gridSize = 24 : i32,
    params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>,
    params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>,
    firstGemmIndices = array<i64: 0>,
    splitKV = 1 : i32,
    storeMethod = #rock<StoreMethod set>,
    operand_segment_sizes = array<i32: 1, 1, 1, 0, 0, 1, 0>
  } : memref<1x64x384xf32>, memref<1x64x384xf32>, memref<1x384x64xf32>, memref<1x384x64xf32>
  return
}

// -----

// CHECK-LABEL: @gridwise_attn_barriers_before_lds_write_issue_1811
func.func @gridwise_attn_barriers_before_lds_write_issue_1811(%arg0: memref<4096xi8>, %arg1: memref<4096xi8>, %arg2: memref<4096xf16>, %arg3: memref<1xi8>, %arg4: memref<1xf16>, %arg5: memref<4096xf16>) attributes {block_size = 64 : i32, grid_size = 1 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK: rock.blockwise_load_tile {{.*}}#rock<GemmLoadTileType BypassLDS>
  // CHECK: affine.for %{{.*}} = 0 to 2 {
  // CHECK: affine.for %{{.*}} = 0 to 1 {
  // CHECK: rock.lds_barrier
  // CHECK: rock.blockwise_load_tile {{.*}}#rock<GemmLoadTileType Default>
  // CHECK: rock.lds_barrier
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: loadAfromLDS
  // CHECK-NOT: loadBfromLDS
  // CHECK: affine.for %{{.*}} = 0 to 2 {
  // CHECK: rock.lds_barrier
  // CHECK: rock.blockwise_load_tile {{.*}}#rock<GemmLoadTileType Default>
  // CHECK: rock.lds_barrier
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: loadAfromLDS
  // CHECK-SAME: loadBfromLDS
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["seq_q", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xi8> to memref<1x64x64xi8>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["seq_k", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xi8> to memref<1x64x64xi8>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["head_v", "seq_k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf16> to memref<1x64x64xf16>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d2)> by [<Unmerge{1} ["seq_k"] at [2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>, <AddDim{1} ["seq_q"] at [1] -> [] at []>] bounds = [1, 1, 1] -> [1]> : memref<1xi8> to memref<1x1x1xi8>
  %4 = rock.transform %arg4 by <affine_map<(d0, d1, d2) -> (d2)> by [<Unmerge{1} ["seq_k"] at [2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>, <AddDim{1} ["seq_q"] at [1] -> [] at []>] bounds = [1, 1, 1] -> [1]> : memref<1xf16> to memref<1x1x1xf16>
  %5 = rock.transform %arg5 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["seq_q", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf16> to memref<1x64x64xf16>
  %6 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 64] -> [1, 64, 64]> : memref<1x64x64xi8> to memref<1x64x64xi8>
  %7 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0N"] at [1, 2] -> ["gemm0K", "gemm0N"] at [2, 1]>] bounds = [1, 64, 64] -> [1, 64, 64]> : memref<1x64x64xi8> to memref<1x64x64xi8>
  %8 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm1K", "gemm1N"] at [1, 2] -> ["gemm1K", "gemm1N"] at [2, 1]>] bounds = [1, 64, 64] -> [1, 64, 64]> : memref<1x64x64xf16> to memref<1x64x64xf16>
  %9 = rock.transform %6 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 64} ["gemm0KPad"] at [1] -> ["gemm0K"] at [1]>, <PassThrough ["gemm0N"] at [2] -> ["gemm0N"] at [2]>] bounds = [1, 128, 64] -> [1, 64, 64]> : memref<1x64x64xi8> to memref<1x128x64xi8>
  %10 = rock.transform %7 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 64} ["gemm0KPad"] at [1] -> ["gemm0K"] at [1]>, <PassThrough ["gemm0M"] at [2] -> ["gemm0M"] at [2]>] bounds = [1, 128, 64] -> [1, 64, 64]> : memref<1x64x64xi8> to memref<1x128x64xi8>
  rock.gridwise_attention_accel(%9, %10, %8, %3, %4, %5) preSoftmaxOps = {
  ^bb0(%arg6: memref<1x64x64xi32>, %arg7: memref<1x1x1xi8>, %arg8: memref<1x1x1xf16>, %arg9: memref<1x64x64xf16>):
    %11 = rock.transform %arg6 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 64} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [64, 64] -> [1, 64, 64]> : memref<1x64x64xi32> to memref<64x64xi32>
    %12 = rock.transform %arg7 by <affine_map<() -> (0, 0, 0)> by [<ConstDim{0, 1} [] at [] -> ["const0"] at [0]>, <ConstDim{0, 1} [] at [] -> ["const1"] at [1]>, <ConstDim{0, 1} [] at [] -> ["const2"] at [2]>] bounds = [] -> [1, 1, 1]> : memref<1x1x1xi8> to memref<i8>
    %13 = rock.transform %arg8 by <affine_map<() -> (0, 0, 0)> by [<ConstDim{0, 1} [] at [] -> ["const0"] at [0]>, <ConstDim{0, 1} [] at [] -> ["const1"] at [1]>, <ConstDim{0, 1} [] at [] -> ["const2"] at [2]>] bounds = [] -> [1, 1, 1]> : memref<1x1x1xf16> to memref<f16>
    %alloc = memref.alloc() : memref<1x64x64xf16>
    %14 = rock.transform %alloc by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{64} ["dim0"] at [0] -> ["exp1"] at [1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <ConstDim{0, 1} [] at [] -> ["unit0"] at [0]>] bounds = [64, 64] -> [1, 64, 64]> : memref<1x64x64xf16> to memref<64x64xf16>
    %15 = rock.transform %12 by <affine_map<(d0, d1) -> ()> by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <AddDim{1} ["exp1"] at [1] -> [] at []>] bounds = [1, 1] -> []> : memref<i8> to memref<1x1xi8>
    %16 = rock.transform %15 by <affine_map<(d0, d1) -> (0, 0)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>] bounds = [64, 64] -> [1, 1]> : memref<1x1xi8> to memref<64x64xi8>
    %17 = rock.transform %13 by <affine_map<(d0, d1) -> ()> by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <AddDim{1} ["exp1"] at [1] -> [] at []>] bounds = [1, 1] -> []> : memref<f16> to memref<1x1xf16>
    %18 = rock.transform %17 by <affine_map<(d0, d1) -> (0, 0)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>] bounds = [64, 64] -> [1, 1]> : memref<1x1xf16> to memref<64x64xf16>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%11, %16, %18 : memref<64x64xi32>, memref<64x64xi8>, memref<64x64xf16>) outs(%14 : memref<64x64xf16>) attrs =  {rock.majorTensorNumber = 0 : index} {
    ^bb0(%in: i32, %in_0: i8, %in_1: f16, %out: f16):
      %19 = arith.extsi %in_0 : i8 to i32
      %20 = arith.subi %in, %19 : i32
      %21 = arith.sitofp %20 : i32 to f16
      %22 = arith.mulf %21, %in_1 : f16
      linalg.yield %22 : f16
    }
    memref.copy %alloc, %arg9 : memref<1x64x64xf16> to memref<1x64x64xf16>
    rock.yield
  } {arch = "amdgcn-amd-amdhsa:gfx1100", blockSize = 64 : i32, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>, gridSize = 1 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 2, 0, 1, 0>, params0 = #rock.wmma_gemm_params<kpackPerBlock = 16, mPerBlock = 32, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, params1 = #rock.wmma_gemm_params<kpackPerBlock = 4, mPerBlock = 32, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>} : memref<1x128x64xi8>, memref<1x128x64xi8>, memref<1x64x64xf16>, memref<1x1x1xi8>, memref<1x1x1xf16>, memref<1x64x64xf16>
  return
}

// -----

// CHECK-LABEL: @gridwise_attn_barriers_before_lds_write_issue_1844
func.func @gridwise_attn_barriers_before_lds_write_issue_1844(%arg0: memref<32768xf16>, %arg1: memref<32768xf16>, %arg2: memref<32768xf16>, %arg3: memref<32768xf16>) attributes {block_size = 256 : i32, grid_size = 2 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
  // CHECK: rock.blockwise_load_tile {{.*}}#rock<GemmLoadTileType BypassLDS>
  // CHECK: affine.for %{{.*}} = 0 to 2 {
  // CHECK: affine.for %{{.*}} = 0 to 1 {
  // CHECK: rock.lds_barrier
  // CHECK: rock.blockwise_load_tile {{.*}}#rock<GemmLoadTileType Default>
  // CHECK: rock.lds_barrier
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: loadAfromLDS
  // CHECK-NOT: loadBfromLDS
  // CHECK: affine.for %{{.*}} = 0 to 1 {
  // CHECK-NOT: rock.lds_barrier
  // CHECK: rock.blockwise_load_tile {{.*}}#rock<GemmLoadTileType Default>
  // CHECK: rock.lds_barrier
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: loadAfromLDS
  // CHECK-NOT: loadBfromLDS
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{256, 128} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]> : memref<32768xf16> to memref<1x256x128xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{256, 128} ["n", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]> : memref<32768xf16> to memref<1x256x128xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{256, 128} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]> : memref<32768xf16> to memref<1x256x128xf16>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{256, 128} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]> : memref<32768xf16> to memref<1x256x128xf16>
  %4 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 128, 256] -> [1, 256, 128]> : memref<1x256x128xf16> to memref<1x128x256xf16>
  %5 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0N"] at [1, 2] -> ["gemm0K", "gemm0N"] at [2, 1]>] bounds = [1, 128, 256] -> [1, 256, 128]> : memref<1x256x128xf16> to memref<1x128x256xf16>
  rock.gridwise_attention_accel(%4, %5, %2, %3) preSoftmaxOps = {
  } {blockSize = 256 : i32, enableSoftmax = false, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>, gridSize = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1, 0>, params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 128, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 128, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>} : memref<1x128x256xf16>, memref<1x128x256xf16>, memref<1x256x128xf16>, memref<1x256x128xf16>
  return
}

// -----

// CHECK-LABEL: @gridwise_attn_barriers_before_lds_write_nobarriers
func.func @gridwise_attn_barriers_before_lds_write_nobarriers(%arg0: memref<16384xf16>, %arg1: memref<16384xf16>, %arg2: memref<16384xf16>, %arg3: memref<16384xf16>) attributes {block_size = 256 : i32, grid_size = 1 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
  // CHECK: rock.blockwise_load_tile {{.*}}#rock<GemmLoadTileType BypassLDS>
  // CHECK: affine.for %{{.*}} = 0 to 1 {
  // CHECK: affine.for %{{.*}} = 0 to 1 {
  // CHECK-NOT: rock.lds_barrier
  // CHECK: rock.blockwise_load_tile {{.*}}#rock<GemmLoadTileType Default>
  // CHECK: rock.lds_barrier
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: loadAfromLDS
  // CHECK-NOT: loadBfromLDS
  // CHECK: affine.for %{{.*}} = 0 to 1 {
  // CHECK-NOT: rock.lds_barrier
  // CHECK: rock.blockwise_load_tile {{.*}}#rock<GemmLoadTileType Default>
  // CHECK: rock.lds_barrier
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: loadAfromLDS
  // CHECK-NOT: loadBfromLDS
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["n", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %4 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 128, 128] -> [1, 128, 128]> : memref<1x128x128xf16> to memref<1x128x128xf16>
  %5 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0N"] at [1, 2] -> ["gemm0K", "gemm0N"] at [2, 1]>] bounds = [1, 128, 128] -> [1, 128, 128]> : memref<1x128x128xf16> to memref<1x128x128xf16>
  rock.gridwise_attention_accel(%4, %5, %2, %3) preSoftmaxOps = {
  } {blockSize = 256 : i32, enableSoftmax = false, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>, gridSize = 1 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1, 0>, params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 128, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 128, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>} : memref<1x128x128xf16>, memref<1x128x128xf16>, memref<1x128x128xf16>, memref<1x128x128xf16>
  return
}

// -----

// CHECK-LABEL: @gridwise_attn_barriers_before_lds_write_nofallback_barrier
func.func @gridwise_attn_barriers_before_lds_write_nofallback_barrier(%arg0: memref<32768xf16>, %arg1: memref<32768xf16>, %arg2: memref<16384xf16>, %arg3: memref<16384xf16>) attributes {block_size = 256 : i32, grid_size = 1 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
  // CHECK: affine.for %{{.*}} = 0 to 1 {
  // CHECK: affine.for %{{.*}} = 0 to 2 {
  // CHECK: rock.lds_barrier
  // CHECK: rock.blockwise_load_tile {{.*}}#rock<GemmLoadTileType DoubleBuffer>
  // CHECK: rock.blockwise_load_tile {{.*}}#rock<GemmLoadTileType Default>
  // CHECK: rock.lds_barrier
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: loadAfromLDS
  // CHECK-NOT: loadBfromLDS
  // CHECK: affine.for %{{.*}} = 0 to 1 {
  // CHECK-NOT: rock.lds_barrier
  // CHECK: rock.blockwise_load_tile {{.*}}#rock<GemmLoadTileType Default>
  // CHECK: rock.lds_barrier
  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: loadAfromLDS
  // CHECK-NOT: loadBfromLDS
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)> by [<Unmerge{128, 256} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 256] -> [32768]> : memref<32768xf16> to memref<1x128x256xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)> by [<Unmerge{128, 256} ["n", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 256] -> [32768]> : memref<32768xf16> to memref<1x128x256xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %4 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 256, 128] -> [1, 128, 256]> : memref<1x128x256xf16> to memref<1x256x128xf16>
  %5 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0N"] at [1, 2] -> ["gemm0K", "gemm0N"] at [2, 1]>] bounds = [1, 256, 128] -> [1, 128, 256]> : memref<1x128x256xf16> to memref<1x256x128xf16>
  rock.gridwise_attention_accel(%4, %5, %2, %3) preSoftmaxOps = {
  } {blockSize = 256 : i32, enableSoftmax = false, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>, gridSize = 1 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1, 0>, params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 128, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 128, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>} : memref<1x256x128xf16>, memref<1x256x128xf16>, memref<1x128x128xf16>, memref<1x128x128xf16>
  return
}
