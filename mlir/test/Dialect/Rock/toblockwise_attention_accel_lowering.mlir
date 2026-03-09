// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -canonicalize -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @gridwise_attn_simple
// CHECK-SAME: (%[[Q:.+]]: memref<1x384x64xf32>, %[[K:.+]]: memref<1x64x384xf32>, %[[V:.+]]: memref<1x384x64xf32>, %[[O:.+]]: memref<1x384x64xf32>)
// CHECK-DAG: %[[ln2Recip:.+]] = arith.constant 1.44269502 : f32
// CHECK-DAG: %[[negInf:.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG: %[[zeroF32:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[zeroVecF32:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>

// CHECK-DAG: %[[QTr0:.+]] = rock.transform %[[Q]] by

// init maxRow buffer
// CHECK-DAG: rock.fill(%[[maxRowBuf:.+]], %[[negInf]]) : memref<1xf32

// init sumRow buffer
// CHECK-DAG: rock.fill(%[[sumRowBuf:.+]], %[[zeroF32]]) : memref<1xf32

// init attentionAcc buffer
// CHECK-DAG: rock.fill(%[[attnOutBuf:.+]], %[[zeroF32]]) : memref<2x16xf32

// Outer N-tile loop
// CHECK: scf.for
  // CHECK-DAG: rock.fill(%[[gemm0AccBuf:.+]], %[[zeroVecF32]])
  // CHECK: %[[ldsG0B:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ldsG0A:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  
  // CHECK: rock.lds_barrier
  // Inner gemm0 KpacksPerBlock loop
  // CHECK: scf.for 
    // CHECK: rock.blockwise_load_tile %[[QTr0]]{{.*}} LDS -> %[[ldsG0B]] -> %[[preAccelRegB:[0-9]+]] {{.*}}#rock<GemmLoadTileType Default>
    // CHECK: rock.blockwise_load_tile %[[K]]{{.*}} LDS -> %[[ldsG0A]] -> %[[preAccelRegA:[0-9]+]] {{.*}}#rock<GemmLoadTileType Default>

    // Emit blockwise gemm0
    // CHECK: rock.stage
    // CHECK: %[[viewG0AStore:.+]] = memref.view %[[ldsG0A]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    // CHECK: %[[viewG0BStore:.+]] = memref.view %[[ldsG0B]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    // CHECK: rock.blockwise_gemm_accel %[[gemm0AccBuf]] += %[[preAccelRegA]] from %[[viewG0AStore]] * %[[preAccelRegB]] from %[[viewG0BStore]]
    // CHECK: {name = "MMA"}
  
  // CHECK: {pipeline = #rock.pipeline<2>}

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
    // CHECK-DAG: %[[newmax:.+]] = arith.maxnumf %[[rowmax]], %[[tilemax]]
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
    // CHECK-DAG: %[[newmax:.+]] = arith.maxnumf %[[rowmax]], %[[tilemax]]
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
  
  // CHECK-DAG: %[[ldsG1BStore:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>

  // Viewing another set of register with kPack packing
  // CHECK: %[[G1AregsKpackTr0:.+]] = rock.transform %[[G1AregsKpack:.+]] by
  // CHECK-DAG: %[[G1AregsKpackTr1:.+]] = rock.transform %[[G1AregsKpackTr0]] by
  // CHECK-DAG: %[[G1AregsKpackTr2:.+]] = rock.transform %[[G1AregsKpackTr1]] by
  // CHECK-DAG: %[[G1AregsKpackTr3:.+]] = rock.transform %[[G1AregsKpackTr2]] by
  // CHECK-DAG: %[[G1AregsKpackTr4:.+]] = rock.transform %[[G1AregsKpackTr3]] by
  // CHECK-DAG: %[[G1AregsKpackTr5:.+]] = rock.transform %[[G1AregsKpackTr4]] by

  // CHECK-DAG: rock.threadwise_copy %[[gemm0NormExpTr5]] -> %[[G1AregsKpackTr5]]

  // Viewing G1 LDS A tile buffer
  // CHECK-DAG: %[[viewG1AStore:.+]] = memref.view %[[ldsG1BStore]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
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
  
  // CHECK-DAG: %[[ldsG0AStore:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>

  // Gemm1
  // CHECK: scf.for %[[g1MIter:.+]]
    // CHECK: rock.blockwise_load_tile %[[V]]{{.*}} LDS -> %[[ldsG0AStore]] -> %[[preAccelRegV:[0-9]+]] {{.*}}#rock<GemmLoadTileType Default>

    // Emit blockwise gemm1
    // rock.stage
    // CHECK-DAG: rock.fill(%[[gemm1AccBuf:.+]], %[[zeroVecF32]])
    // CHECK: %[[view2G1AStore:.+]] = memref.view %[[ldsG0AStore]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    // CHECK: %[[view2G1BStore:.+]] = memref.view %[[ldsG1BStore]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    // CHECK: rock.blockwise_gemm_accel %[[gemm1AccBuf]] += %[[preAccelRegV]] from %[[view2G1AStore]] * %[[preAccelRegA:[0-9]+]] from %[[view2G1BStore]]
    // CHECK: {name = "MMA"}

    // rock.stage
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
    // CHECK: {name = "PostProcess"}
  // CHECK : {pipeline = #rock.pipeline<2>}
// CHECK : }
// CHECK : %[[flatAttnOutBuf:.+]] = memref.collapse_shape %[[attnOutBuf]]
// CHECK : rock.threadwise_write_all {{.*}} %[[flatAttnOutBuf]] -> {{.*}}(%[[O]])

func.func @gridwise_attn_simple(%arg0: memref<1x384x64xf32>, %arg1: memref<1x64x384xf32>, %arg2: memref<1x384x64xf32>, %arg3: memref<1x384x64xf32>) attributes {block_size = 64 : i32, grid_size = 24 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>
  rock.gridwise_attention_accel(%0, %arg1, %arg2, %arg3) preSoftmaxOps = {} {
    blockSize = 64 : i32,
    gridSize = 24 : i32,
    params0 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>,
    params1 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>,
    firstGemmIndices = array<i64: 0>,
    splitKV = 1 : i32,
    storeMethod = #rock<StoreMethod set>,
    operand_segment_sizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 0>
  } : memref<1x64x384xf32>, memref<1x64x384xf32>, memref<1x384x64xf32>, memref<1x384x64xf32>
  return
}

// CHECK-LABEL: @gridwise_attn_schedulev2
func.func @gridwise_attn_schedulev2(%arg0: memref<1x384x64xf32>, %arg1: memref<1x64x384xf32>, %arg2: memref<1x384x64xf32>, %arg3: memref<1x384x64xf32>) attributes {block_size = 64 : i32, grid_size = 24 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>

  // CHECK: scf.for
  // CHECK: rock.lds_barrier
  // CHECK: scf.for

  // CHECK: rock.blockwise_load_tile
  // CHECK: loadType = #rock<GemmLoadTileType DoubleBuffer>

  // CHECK: rock.blockwise_load_tile
  // CHECK: loadType = #rock<GemmLoadTileType DoubleBuffer>

  // CHECK: rock.stage
  // CHECK: rock.blockwise_gemm_accel 
  // CHECK-NOT: loadAfromLDS
  // CHECK-NOT: loadBfromLDS
  // CHECK: {name = "MMA"}

  // scf.for

  // CHECK: rock.blockwise_load_tile
  // CHECK: loadType = #rock<GemmLoadTileType DoubleBuffer>

  // CHECK: rock.stage
  // CHECK: rock.blockwise_gemm_accel
  // CHECK: {name = "MMA"}
  rock.gridwise_attention_accel(%0, %arg1, %arg2, %arg3) preSoftmaxOps = {} {
    blockSize = 64 : i32,
    gridSize = 24 : i32,
    params0 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>,
    params1 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>,
    firstGemmIndices = array<i64: 0>,
    splitKV = 1 : i32,
    storeMethod = #rock<StoreMethod set>,
    operand_segment_sizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 0>
  } : memref<1x64x384xf32>, memref<1x64x384xf32>, memref<1x384x64xf32>, memref<1x384x64xf32>
  return
}

// Test that we are properly initializing output buffers when partial early exit
// is enabled.
#accel_gemm_params = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>
#map = affine_map<(d0, d1, d2, d3) -> (d2 * 12288 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 2 + d2) * 512 + d3) * 128 + d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 * 128 + d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d1, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>
#map7 = affine_map<(d0, d1, d2) -> (0, d0 floordiv 2, d0 mod 2, d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> ((d0 * 512 + d1) * 128 + d2)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> ((((d0 * 2 + d1) * 5 + d2) * 32 + d3) * 128 + d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 32 + d1) * 2 + d2) * 5 + d3 + d4)>
#map11 = affine_map<(d0, d1) -> (0, d0 floordiv 2, d0 mod 2, d1, 0)>
#map12 = affine_map<(d0, d1, d2) -> (d2)>
#map13 = affine_map<(d0, d1, d2) -> (d0, 0, 0)>
#map14 = affine_map<(d0) -> (0, d0 floordiv 2, d0 mod 2)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d1, d2, d3)>
#map16 = affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)>
#map17 = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 512, d1, d2 mod 512)>
#map18 = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 512, d1 mod 512, d2)>
#map19 = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#map20 = affine_map<(d0) -> (d0, 0)>
#map21 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map22 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map23 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map24 = affine_map<(d0, d1) -> (d0, d1)>
#map25 = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 2 + d2, d3, d4)>
#map26 = affine_map<(d0, d1, d2, d3) -> (0, d0, d1, d2, d3)>
#map27 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#transform_map = #rock.transform_map<#map by [<Unmerge{5, 12288} ["exp2", "exp3"] at [2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit1"] at [1] -> [] at []>] bounds = [1, 1, 5, 12288] -> [61440]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 2, 5, 12288] -> [1, 1, 5, 12288]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{32, 2, 512, 128} ["exp1", "exp2", "exp3", "exp4"] at [1, 2, 3, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 32, 2, 512, 128] -> [4194304]>
#transform_map3 = #rock.transform_map<#map3 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <Unmerge{96, 128} ["exp3", "exp4"] at [3, 4] -> ["dim3"] at [3]>] bounds = [1, 2, 5, 96, 128] -> [1, 2, 5, 12288]>
#transform_map4 = #rock.transform_map<#map4 by [<PassThrough ["dim0", "dim3", "dim1", "dim2", "dim4"] at [0, 1, 2, 3, 4] -> ["dim0", "dim3", "dim1", "dim2", "dim4"] at [0, 3, 1, 2, 4]>] bounds = [1, 96, 2, 5, 128] -> [1, 2, 5, 96, 128]>
#transform_map5 = #rock.transform_map<#map5 by [<Slice{0, 1, 0, 32, 0, 2, 0, 5, 0, 128} ["dim0_sliced", "dim1_sliced", "dim2_sliced", "dim3_sliced", "dim4_sliced"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim2", "dim3", "dim4"] at [0, 1, 2, 3, 4]>] bounds = [1, 32, 2, 5, 128] -> [1, 96, 2, 5, 128]>
#transform_map6 = #rock.transform_map<#map6 by [<PassThrough ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 4, 3]>] bounds = [1, 32, 2, 128, 512] -> [1, 32, 2, 512, 128]>
#transform_map7 = #rock.transform_map<#map7 by [<Merge{1, 32, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [64, 5, 128] -> [1, 32, 2, 5, 128]>
#transform_map8 = #rock.transform_map<#map7 by [<Merge{1, 32, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [64, 128, 512] -> [1, 32, 2, 128, 512]>
#transform_map9 = #rock.transform_map<#map8 by [<Unmerge{64, 512, 128} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [64, 512, 128] -> [4194304]>
#transform_map10 = #rock.transform_map<#map9 by [<Unmerge{1, 2, 5, 32, 128} ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4] -> ["dim0"] at [0]>] bounds = [1, 2, 5, 32, 128] -> [40960]>
#transform_map11 = #rock.transform_map<#map4 by [<PassThrough ["dim0", "dim2", "dim3", "dim1", "dim4"] at [0, 2, 3, 1, 4] -> ["dim0", "dim2", "dim3", "dim1", "dim4"] at [0, 1, 2, 3, 4]>] bounds = [1, 32, 2, 5, 128] -> [1, 2, 5, 32, 128]>
#transform_map12 = #rock.transform_map<#map7 by [<Merge{32, 2} ["dim0"] at [0] -> ["exp1", "exp2"] at [1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>, <ConstDim{0, 1} [] at [] -> ["unit0"] at [0]>] bounds = [64, 5, 128] -> [1, 32, 2, 5, 128]>
#transform_map13 = #rock.transform_map<#map10 by [<Unmerge{1, 32, 2, 5, 1} ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4] -> ["dim0"] at [0]>] bounds = [1, 32, 2, 5, 1] -> [320]>
#transform_map14 = #rock.transform_map<#map11 by [<Merge{32, 2} ["dim0"] at [0] -> ["exp1", "exp2"] at [1, 2]>, <Merge{5} ["dim1"] at [1] -> ["exp3"] at [3]>, <ConstDim{0, 1} [] at [] -> ["unit0"] at [0]>, <ConstDim{0, 1} [] at [] -> ["unit4"] at [4]>] bounds = [64, 5] -> [1, 32, 2, 5, 1]>
#transform_map15 = #rock.transform_map<#map12 by [<Unmerge{1} ["exp2"] at [2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit1"] at [1] -> [] at []>] bounds = [1, 1, 1] -> [1]>
#transform_map16 = #rock.transform_map<#map13 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [1, 32, 2] -> [1, 1, 1]>
#transform_map17 = #rock.transform_map<#map14 by [<Merge{1, 32, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [64] -> [1, 32, 2]>
#transform_map18 = #rock.transform_map<#map15 by [<Unmerge{32, 2} ["batch", "splitKV"] at [0, 1] -> ["batch_merged"] at [0]>, <PassThrough ["M", "K"] at [2, 3] -> ["M", "K"] at [1, 2]>] bounds = [32, 2, 5, 128] -> [64, 5, 128]>
#transform_map19 = #rock.transform_map<#map16 by [<PassThrough ["batch"] at [0] -> ["batch"] at [0]>, <ConstDim{0, 2} [] at [] -> ["splitKV"] at [1]>, <PassThrough ["M", "K"] at [1, 2] -> ["M", "K"] at [2, 3]>] bounds = [32, 5, 128] -> [32, 2, 5, 128]>
#transform_map20 = #rock.transform_map<#map15 by [<Unmerge{32, 2} ["batch", "splitKV"] at [0, 1] -> ["batch_merged"] at [0]>, <PassThrough ["K", "seq_k_chunk"] at [2, 3] -> ["K", "seq_k_chunk"] at [1, 2]>] bounds = [32, 2, 128, 512] -> [64, 128, 512]>
#transform_map21 = #rock.transform_map<#map17 by [<PassThrough ["batch"] at [0] -> ["batch"] at [0]>, <Merge{2, 512} ["seq_k"] at [2] -> ["splitKV", "seq_k_chunk"] at [1, 3]>, <PassThrough ["K"] at [1] -> ["K"] at [2]>] bounds = [32, 128, 1024] -> [32, 2, 128, 512]>
#transform_map22 = #rock.transform_map<#map15 by [<Unmerge{32, 2} ["batch", "splitKV"] at [0, 1] -> ["batch_merged"] at [0]>, <PassThrough ["seq_k_chunk", "D"] at [2, 3] -> ["seq_k_chunk", "D"] at [1, 2]>] bounds = [32, 2, 512, 128] -> [64, 512, 128]>
#transform_map23 = #rock.transform_map<#map18 by [<PassThrough ["batch"] at [0] -> ["batch"] at [0]>, <Merge{2, 512} ["seq_k"] at [1] -> ["splitKV", "seq_k_chunk"] at [1, 2]>, <PassThrough ["D"] at [2] -> ["D"] at [3]>] bounds = [32, 1024, 128] -> [32, 2, 512, 128]>
#transform_map24 = #rock.transform_map<#map19 by [<Unmerge{32, 2} ["batch", "splitKV"] at [0, 1] -> ["batch_merged"] at [0]>] bounds = [32, 2] -> [64]>
#transform_map25 = #rock.transform_map<#map20 by [<PassThrough ["batch"] at [0] -> ["batch"] at [0]>, <ConstDim{0, 2} [] at [] -> ["splitKV"] at [1]>] bounds = [32] -> [32, 2]>
#transform_map26 = #rock.transform_map<#map21 by [<PassThrough ["dim1", "dim0", "dim2"] at [0, 1, 2] -> ["dim1", "dim0", "dim2"] at [1, 0, 2]>] bounds = [5, 32, 128] -> [32, 5, 128]>
#transform_map27 = #rock.transform_map<#map21 by [<PassThrough ["dim1", "dim0", "dim2"] at [0, 1, 2] -> ["dim1", "dim0", "dim2"] at [1, 0, 2]>] bounds = [32, 5, 128] -> [5, 32, 128]>
#transform_map28 = #rock.transform_map<#map22 by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [32, 1024, 128] -> [32, 128, 1024]>
#transform_map29 = #rock.transform_map<#map22 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [32, 128, 5] -> [32, 5, 128]>
#transform_map30 = #rock.transform_map<#map22 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0N"] at [1, 2] -> ["gemm0K", "gemm0N"] at [2, 1]>] bounds = [32, 128, 1024] -> [32, 1024, 128]>
#transform_map31 = #rock.transform_map<#map23 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K"] at [1] -> ["gemm0K"] at [1]>, <Pad{0, 27} ["gemm0NPad"] at [2] -> ["gemm0N"] at [2]>] bounds = [32, 128, 32] -> [32, 128, 5]>
#transform_map32 = #rock.transform_map<#map23 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 27} ["gemm1NPad"] at [1] -> ["gemm1N"] at [1]>, <PassThrough ["gemm1M"] at [2] -> ["gemm1M"] at [2]>] bounds = [64, 32, 128] -> [64, 5, 128]>
#transform_map33 = #rock.transform_map<#map24 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 27} ["gemm1NPad"] at [1] -> ["gemm1N"] at [1]>] bounds = [64, 32] -> [64, 5]>
#transform_map34 = #rock.transform_map<#map25 by [<Unmerge{32, 2} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [3] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [4] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 32, 2, 5, 512] -> [64, 5, 512]>
#transform_map35 = #rock.transform_map<#map26 by [<Merge{1, 32} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [4]>] bounds = [32, 2, 5, 512] -> [1, 32, 2, 5, 512]>
#transform_map36 = #rock.transform_map<#map26 by [<Merge{32} ["dim0"] at [0] -> ["exp1"] at [1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [4]>, <ConstDim{0, 1} [] at [] -> ["unit0"] at [0]>] bounds = [32, 2, 5, 512] -> [1, 32, 2, 5, 512]>
// CHECK-LABEL: @mlir_attention
// CHECK-DAG: %[[negInfF32:.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG: %[[zeroF16:.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG: %[[zeroF32:.+]] = arith.constant 0.000000e+00 : f32

// init lseBuffer to -inf (for early exit case)
// CHECK: rock.fill(%[[lseBuf:.+]], %[[negInfF32]]) : memref<16xf32, #gpu.address_space<private>>, f32

// init outAccBufferOutTyped (f16) to 0 (for early exit case when types differ)
// CHECK: rock.fill(%[[outAccBufF16:.+]], %[[zeroF16]]) : memref<4x16xf16, #gpu.address_space<private>>, f16

// init attentionOutAccBuffer (f32) to 0
// CHECK: rock.fill(%[[attnOutAccBuf:.+]], %[[zeroF32]]) : memref<4x16xf32, #gpu.address_space<private>>, f32

// early exit conditional block
// CHECK: scf.if
func.func @mlir_attention(%arg0: memref<61440xf16>, %arg1: memref<4194304xf16>, %arg2: memref<1xi32>, %arg3: memref<4194304xf16>, %arg4: memref<40960xf16>, %arg5: memref<320xf32>) attributes {arch = "gfx950", block_size = 64 : i32, features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b>, grid_size = 64 : i32, kernel = "mixr", num_cu = 256 : i64} {
  %cst = arith.constant 8.837890e-02 : f16
  %0 = rock.transform %arg0 by #transform_map : memref<61440xf16> to memref<1x1x5x12288xf16>
  %1 = rock.transform %0 by #transform_map1 : memref<1x1x5x12288xf16> to memref<1x2x5x12288xf16>
  %2 = rock.transform %arg1 by #transform_map2 : memref<4194304xf16> to memref<1x32x2x512x128xf16>
  %3 = rock.transform %1 by #transform_map3 : memref<1x2x5x12288xf16> to memref<1x2x5x96x128xf16>
  %4 = rock.transform %3 by #transform_map4 : memref<1x2x5x96x128xf16> to memref<1x96x2x5x128xf16>
  %5 = rock.transform %4 by #transform_map5 : memref<1x96x2x5x128xf16> to memref<1x32x2x5x128xf16>
  %6 = rock.transform %2 by #transform_map6 : memref<1x32x2x512x128xf16> to memref<1x32x2x128x512xf16>
  %7 = rock.transform %5 by #transform_map7 : memref<1x32x2x5x128xf16> to memref<64x5x128xf16>
  %8 = rock.transform %6 by #transform_map8 : memref<1x32x2x128x512xf16> to memref<64x128x512xf16>
  %9 = rock.transform %arg3 by #transform_map9 : memref<4194304xf16> to memref<64x512x128xf16>
  %alloc = memref.alloc() : memref<40960xf16>
  %10 = rock.transform %alloc by #transform_map10 : memref<40960xf16> to memref<1x2x5x32x128xf16>
  %11 = rock.transform %10 by #transform_map11 : memref<1x2x5x32x128xf16> to memref<1x32x2x5x128xf16>
  %12 = rock.transform %11 by #transform_map12 : memref<1x32x2x5x128xf16> to memref<64x5x128xf16>
  %alloc_0 = memref.alloc() : memref<320xf32>
  %13 = rock.transform %alloc_0 by #transform_map13 : memref<320xf32> to memref<1x32x2x5x1xf32>
  %14 = rock.transform %13 by #transform_map14 : memref<1x32x2x5x1xf32> to memref<64x5xf32>
  %15 = rock.transform %arg2 by #transform_map15 : memref<1xi32> to memref<1x1x1xi32>
  %16 = rock.transform %15 by #transform_map16 : memref<1x1x1xi32> to memref<1x32x2xi32>
  %17 = rock.transform %16 by #transform_map17 : memref<1x32x2xi32> to memref<64xi32>
  %18 = rock.transform %7 by #transform_map18 : memref<64x5x128xf16> to memref<32x2x5x128xf16>
  %19 = rock.transform %18 by #transform_map19 : memref<32x2x5x128xf16> to memref<32x5x128xf16>
  %20 = rock.transform %8 by #transform_map20 : memref<64x128x512xf16> to memref<32x2x128x512xf16>
  %21 = rock.transform %20 by #transform_map21 : memref<32x2x128x512xf16> to memref<32x128x1024xf16>
  %22 = rock.transform %9 by #transform_map22 : memref<64x512x128xf16> to memref<32x2x512x128xf16>
  %23 = rock.transform %22 by #transform_map23 : memref<32x2x512x128xf16> to memref<32x1024x128xf16>
  %24 = rock.transform %17 by #transform_map24 : memref<64xi32> to memref<32x2xi32>
  %25 = rock.transform %24 by #transform_map25 : memref<32x2xi32> to memref<32xi32>
  %26 = rock.transform %19 by #transform_map26 : memref<32x5x128xf16> to memref<5x32x128xf16>
  %27 = rock.transform %26 by #transform_map27 : memref<5x32x128xf16> to memref<32x5x128xf16>
  %28 = rock.transform %21 by #transform_map28 : memref<32x128x1024xf16> to memref<32x1024x128xf16>
  %29 = rock.transform %27 by #transform_map29 : memref<32x5x128xf16> to memref<32x128x5xf16>
  %30 = rock.transform %28 by #transform_map30 : memref<32x1024x128xf16> to memref<32x128x1024xf16>
  %31 = rock.transform %29 by #transform_map31 : memref<32x128x5xf16> to memref<32x128x32xf16>
  %32 = rock.transform %12 by #transform_map32 : memref<64x5x128xf16> to memref<64x32x128xf16>
  %33 = rock.transform %14 by #transform_map33 : memref<64x5xf32> to memref<64x32xf32>
  rock.gridwise_attention_accel(%31, %30, %23, %25, %32, %33) preSoftmaxOps = {
  ^bb0(%arg6: memref<64x5x512xf16>, %arg7: memref<1x32x2x5x512xf16>):
    %34 = rock.transform %arg6 by #transform_map34 : memref<64x5x512xf16> to memref<1x32x2x5x512xf16>
    %35 = rock.transform %34 by #transform_map35 : memref<1x32x2x5x512xf16> to memref<32x2x5x512xf16>
    %alloc_1 = memref.alloc() : memref<1x32x2x5x512xf16>
    %36 = rock.transform %alloc_1 by #transform_map36 : memref<1x32x2x5x512xf16> to memref<32x2x5x512xf16>
    linalg.generic {indexing_maps = [#map27, #map27], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35 : memref<32x2x5x512xf16>) outs(%36 : memref<32x2x5x512xf16>) attrs =  {rock.majorTensorNumber = 0 : index} {
    ^bb0(%in: f16, %out: f16):
      %37 = arith.mulf %in, %cst : f16
      linalg.yield %37 : f16
    }
    memref.copy %alloc_1, %arg7 : memref<1x32x2x5x512xf16> to memref<1x32x2x5x512xf16>
    rock.yield
  } {blockSize = 64 : i32, causal, firstGemmIndices = array<i64: 0>, gridSize = 64 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 1, 1>, params0 = #accel_gemm_params, params1 = #accel_gemm_params, prePadG0N = 5 : index, preSoftmaxHasSplitKVTransforms = true, softmaxType = f32, splitKV = 2 : i32, storeMethod = #rock<StoreMethod set>} : memref<32x128x32xf16>, memref<32x128x1024xf16>, memref<32x1024x128xf16>, memref<32xi32>, memref<64x32x128xf16>, memref<64x32xf32>
  memref.copy %alloc, %arg4 : memref<40960xf16> to memref<40960xf16>
  memref.copy %alloc_0, %arg5 : memref<320xf32> to memref<320xf32>
  return
}

