// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -canonicalize %s | FileCheck %s

#xdlops_gemm_params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 32, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor=1, scheduleVersion=1, outputSwizzle=2, forceUnroll = true>
// CHECK-LABEL: @gridwise_attn_simple
// CHECK-SAME: (%[[Q:.+]]: memref<1x384x64xf32>, %[[K:.+]]: memref<1x64x384xf32>, %[[V:.+]]: memref<1x384x64xf32>, %[[O:.+]]: memref<1x384x64xf32>)
// CHECK-DAG: %[[ln2Recip:.+]] = arith.constant 1.44269502 : f32
// CHECK-DAG: %[[negInf:.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG: %[[zeroF32:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[zeroVecF32:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>

// CHECK: %[[QTr0:.+]] = rock.transform %[[Q]] by

// init maxRow buffer
// CHECK-DAG: rock.fill(%[[maxRowBuf:.+]], %[[negInf]])

// init sumRow buffer
// CHECK-DAG: rock.fill(%[[sumRowBuf:.+]], %[[zeroF32]])

// init attentionAcc buffer
// CHECK-DAG: rock.fill(%[[attnOutBuf:.+]], %[[zeroF32]])

// Outer N-tile loop
// CHECK: affine.for
  // CHECK-DAG: rock.fill(%[[gemm0AccBuf:.+]], %[[zeroVecF32]])

  // Inner gemm0 KpacksPerBlock loop
  // CHECK: affine.for
    // CHECK: %[[ldsG0A:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
    // Load G0A tile to regs
    // CHECK-DAG: %[[QTr1:.+]] = rock.transform %[[QTr0]] by
    // CHECK-DAG: %[[QTr2:.+]] = rock.transform %[[QTr1]] by
    // CHECK-DAG: rock.threadwise_read_into {{.*}}(%[[QTr2]]) {{.*}} -> %[[G0Aregs:.+]] :

    // Repack G0A tile regs for better LDS store vectorization
    // CHECK-DAG: %[[G0AregsTr0:.+]] = rock.transform %[[G0Aregs]] by
    // CHECK-DAG: %[[G0AregsTr1:.+]] = rock.transform %[[G0AregsTr0]] by
    // CHECK: %[[G0AregsKpackTr0:.+]] = rock.transform %[[G0AregsKpack:.+]] by
    // CHECK-DAG: %[[G0AregsKpackTr1:.+]] = rock.transform %[[G0AregsKpackTr0:.+]] by
    // CHECK-DAG: rock.threadwise_copy %[[G0AregsTr1]] -> %[[G0AregsKpackTr1]]

    // Viewing G0 LDS A tile buffer
    // CHECK-DAG: %[[viewG0AStore:.+]] = memref.view %[[ldsG0A]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    // CHECK-DAG: %[[viewG0AStoreTr0:.+]] = rock.transform %[[viewG0AStore]]
    // CHECK-DAG: %[[viewG0AStoreTr1:.+]] = rock.transform %[[viewG0AStoreTr0]]
    // CHECK-DAG: %[[viewG0AStoreTr2:.+]] = rock.transform %[[viewG0AStoreTr1]]
    // CHECK-DAG: %[[viewG0AStoreTr3:.+]] = rock.transform %[[viewG0AStoreTr2]]

    // Store to LDS G0A tile buffer
    // CHECK-DAG: rock.threadwise_write_all {{.*}} %[[G0AregsKpack]] -> [](%[[viewG0AStoreTr3]])
    // CHECK-DAG: %[[view2G0AStore:.+]] = memref.view %[[ldsG0A]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    // CHECK: %[[ldsG0B:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>

    // Load G0B tile to regs
    // CHECK-DAG: %[[KTr0:.+]] = rock.transform %[[K]] by
    // CHECK-DAG: %[[KTr1:.+]] = rock.transform %[[KTr0]] by
    // CHECK-DAG: rock.threadwise_read_into {{.*}}(%[[KTr1]]) {{.*}} -> %[[G0Bregs:.+]] :

    // Repack G0B tile regs for better LDS store vectorization
    // CHECK-DAG: %[[G0BregsTr0:.+]] = rock.transform %[[G0Bregs]] by
    // CHECK-DAG: %[[G0BregsTr1:.+]] = rock.transform %[[G0BregsTr0]] by
    // CHECK: %[[G0BregsKpackTr0:.+]] = rock.transform %[[G0BregsKpack:.+]] by
    // CHECK-DAG: %[[G0BregsKpackTr1:.+]] = rock.transform %[[G0BregsKpackTr0:.+]] by
    // CHECK-DAG: rock.threadwise_copy %[[G0BregsTr1]] -> %[[G0BregsKpackTr1]]

    // Viewing G0 LDS B tile buffer
    // CHECK-DAG: %[[viewG0BStore:.+]] = memref.view %[[ldsG0B]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    // CHECK-DAG: %[[viewG0BStoreTr0:.+]] = rock.transform %[[viewG0BStore]]
    // CHECK-DAG: %[[viewG0BStoreTr1:.+]] = rock.transform %[[viewG0BStoreTr0]]
    // CHECK-DAG: %[[viewG0BStoreTr2:.+]] = rock.transform %[[viewG0BStoreTr1]]
    // CHECK-DAG: %[[viewG0BStoreTr3:.+]] = rock.transform %[[viewG0BStoreTr2]]

    // Store to LDS G0B tile buffer
    // CHECK-DAG: rock.threadwise_write_all {{.*}} %[[G0BregsKpack]] -> [](%[[viewG0BStoreTr3]])
    // CHECK-DAG: %[[view2G0BStore:.+]] = memref.view %[[ldsG0B]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    // CHECK: rock.lds_barrier

    // Load G0A from LDS to regs
    // CHECK-DAG: %[[view2G0AStoreTr0:.+]] = rock.transform %[[view2G0AStore]]
    // CHECK-DAG: %[[view2G0AStoreTr1:.+]] = rock.transform %[[view2G0AStoreTr0]]
    // CHECK-DAG: %[[view2G0AStoreTr2:.+]] = rock.transform %[[view2G0AStoreTr1]]
    // CHECK-DAG: %[[view2G0AStoreTr3:.+]] = rock.transform %[[view2G0AStoreTr2]]
    // CHECK: affine.for
      // CHECK: rock.threadwise_read_into {{.*}} [](%[[view2G0AStoreTr3]]) {{.*}} -> %[[preAccelRegA:.+]] :
    // CHECK: rock.dealloc %[[ldsG0A]] : memref<4096xi8, #gpu.address_space<workgroup>>

    // Load G0B from LDS to regs and accel gemm
    // CHECK-DAG: %[[view2G0BStoreTr0:.+]] = rock.transform %[[view2G0BStore]]
    // CHECK-DAG: %[[view2G0BStoreTr1:.+]] = rock.transform %[[view2G0BStoreTr0]]
    // CHECK-DAG: %[[view2G0BStoreTr2:.+]] = rock.transform %[[view2G0BStoreTr1]]
    // CHECK-DAG: %[[view2G0BStoreTr3:.+]] = rock.transform %[[view2G0BStoreTr2]]
    // CHECK: affine.for
      // CHECK: affine.for
        // CHECK: rock.threadwise_read_into {{.*}} [](%[[view2G0BStoreTr3]]) {{.*}} -> %[[preAccelRegB:.+]] :
        // CHECK: rock.dealloc %[[ldsG0B]] : memref<4096xi8, #gpu.address_space<workgroup>>
        // CHECK: %[[bufferA:.*]] = rock.transform %[[preAccelRegB]]
        // CHECK: %[[bufferB:.*]] = rock.transform %[[preAccelRegA]]
        // CHECK: %[[bufferC:.*]] = rock.transform %[[gemm0AccBuf]]
        // CHECK: rock.threadwise_accel_gemm %[[bufferC]]{{.*}} += %[[bufferA:.*]] * %[[bufferB:.*]]

  // End of inner gemm0 KpacksPerBlock loop
  // CHECK: }
  // CHECK: rock.transforming_for
    // CHECK: %[[tmp:.+]] =  memref.load %[[gemm0AccBuf]][
    // CHECK: rock.in_bounds_store %[[tmp]] -> %[[gemm0AccBufScalar:.+]][
  // CHECK: linalg.generic {{.*}} ins(%[[gemm0AccBufScalar]] {{.*}} outs(%[[gemm0AccBufScalar]]
    // CHECK: %[[gemm0Scaled:.+]] = arith.mulf %in, %[[ln2Recip]] : f32
    // CHECK: linalg.yield %[[gemm0Scaled]]
  // CHECK: %[[ldsReductionWS:.+]] = rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ldsReductionWSView:.+]] = memref.view %[[ldsReductionWS]][{{.*}}][] : memref<256xi8, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
  // CHECK: rock.blockwise_broadcast_reduce max {{.*}} %[[gemm0AccBufScalar]] into %[[gemm0Max:[0-9]+]] using %[[ldsReductionWSView]]
  // CHECK: rock.dealloc %[[ldsReductionWS]] : memref<256xi8, #gpu.address_space<workgroup>>

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
  // CHECK: rock.dealloc %[[ldsReductionWS2]] : memref<256xi8, #gpu.address_space<workgroup>>

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
  
  // Viewing LDS G1A tile buffer in MFMA layout
  // CHECK-DAG: %[[viewG1ALoadTr0:.+]] = rock.transform %[[view2G1AStore]]
  // CHECK-DAG: %[[viewG1ALoadTr1:.+]] = rock.transform %[[viewG1ALoadTr0]]
  // CHECK-DAG: %[[viewG1ALoadTr2:.+]] = rock.transform %[[viewG1ALoadTr1]]
  // CHECK-DAG: %[[viewG1ALoadTr3:.+]] = rock.transform %[[viewG1ALoadTr2]]

  // Gemm1
  // CHECK: affine.for %[[g1MIter:.+]]
    // CHECK-DAG: rock.fill(%[[gemm1AccBuf:.+]], %[[zeroVecF32]])
    // CHECK-DAG: %[[ldsG0BStore:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>

    // Load G1B tile from global to regs
    // CHECK-DAG: %[[VTr0:.+]] = rock.transform %[[V]] by
    // CHECK-DAG: %[[VTr1:.+]] = rock.transform %[[VTr0]] by
    // CHECK-DAG: rock.threadwise_read_into {{.*}}(%[[VTr1]]) {{.*}} -> %[[G1Bregs:.+]] :

    // Repack G1B tile regs for better LDS store vectorization
    // CHECK-DAG: %[[G1BregsTr0:.+]] = rock.transform %[[G1Bregs]] by
    // CHECK-DAG: %[[G1BregsTr1:.+]] = rock.transform %[[G1BregsTr0]] by
    // CHECK: %[[G1BregsKpackTr0:.+]] = rock.transform %[[G1BregsKpack:.+]] by
    // CHECK-DAG: %[[G1BregsKpackTr1:.+]] = rock.transform %[[G1BregsKpackTr0:.+]] by
    // CHECK-DAG: rock.threadwise_copy %[[G1BregsTr1]] -> %[[G1BregsKpackTr1]]

    // Viewing G1 LDS B tile buffer
    // CHECK-DAG: %[[viewG1BStore:.+]] = memref.view %[[ldsG0BStore]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    // CHECK-DAG: %[[viewG1BStoreTr0:.+]] = rock.transform %[[viewG1BStore]]
    // CHECK-DAG: %[[viewG1BStoreTr1:.+]] = rock.transform %[[viewG1BStoreTr0]]
    // CHECK-DAG: %[[viewG1BStoreTr2:.+]] = rock.transform %[[viewG1BStoreTr1]]
    // CHECK-DAG: %[[viewG1BStoreTr3:.+]] = rock.transform %[[viewG1BStoreTr2]]

    // Store to LDS G1B tile buffer
    // CHECK-DAG: rock.threadwise_write_all {{.*}} %[[G1BregsKpack]] -> [](%[[viewG1BStoreTr3]])
    // CHECK-DAG: %[[view2G1BStore:.+]] = memref.view %[[ldsG0BStore]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>

    // Viewing LDS G1B tile buffer in MFMA layout
    // CHECK-DAG: %[[viewG1BLoadTr0:.+]] = rock.transform %[[view2G1BStore]]
    // CHECK-DAG: %[[viewG1BLoadTr1:.+]] = rock.transform %[[viewG1BLoadTr0]]
    // CHECK-DAG: %[[viewG1BLoadTr2:.+]] = rock.transform %[[viewG1BLoadTr1]]
    // CHECK-DAG: %[[viewG1BLoadTr3:.+]] = rock.transform %[[viewG1BLoadTr2]]

    // CHECK-DAG: rock.lds_barrier
    // CHECK: affine.for
        // CHECK: affine.for
          // CHECK: rock.threadwise_read_into {{.*}} [](%[[viewG1BLoadTr3]]) {{.*}} -> %[[preAccelRegB:.+]] :
          // CHECK: rock.dealloc %[[ldsG0BStore]] : memref<4096xi8, #gpu.address_space<workgroup>>
          // CHECK: rock.threadwise_read_into {{.*}} [](%[[viewG1ALoadTr3]]) {{.*}} -> %[[preAccelRegA:.+]] :
          // CHECK: rock.dealloc %[[ldsG1AStore]] : memref<4096xi8, #gpu.address_space<workgroup>>
          // CHECK: %[[bufferA:.*]] = rock.transform %[[preAccelRegB]]
          // CHECK: %[[bufferB:.*]] = rock.transform %[[preAccelRegA]]
          // CHECK: %[[bufferC:.*]] = rock.transform %[[gemm1AccBuf]]
          // CHECK: rock.threadwise_accel_gemm %[[bufferC]]{{.*}} += %[[bufferA:.*]] * %[[bufferB:.*]]

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
  rock.gridwise_attention_accel(%0, %arg1, %arg2, %arg3) features =  mfma|dot|atomic_add|atomic_add_f16 preSoftmaxOps = {} {
    arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-",
    blockSize = 64 : i32,
    gridSize = 24 : i32,
    params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>,
    params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>,
    firstGemmIdx = 0 : i32,
    operand_segment_sizes = array<i32: 1, 1, 1, 0, 0, 1>
  } : memref<1x64x384xf32>, memref<1x64x384xf32>, memref<1x384x64xf32>, memref<1x384x64xf32>
  return
}

// -----

// CHECK-DAG: #[[REV_MAP_G0M:.+]] = affine_map<(d0) -> (-d0 + 11)>
// CHECK-DAG: #[[REV_MAP_G0K:.+]] = affine_map<(d0) -> (-d0 + 1)>
// CHECK: @gridwise_attn_grid_reversed
func.func @gridwise_attn_grid_reversed(%arg0: memref<1x384x64xf32>, %arg1: memref<1x64x384xf32>, %arg2: memref<1x384x64xf32>, %arg3: memref<1x384x64xf32>) attributes {block_size = 64 : i32, grid_size = 24 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-", reverse_grid} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>
  // CHECK: affine.for %[[MITER:.+]] = 0 to 12 {
    // CHECK: %[[REV_MITER:.+]] = affine.apply #[[REV_MAP_G0M]](%[[MITER]])
    // CHECK: affine.for %[[G0KITER:.+]] = 0 to 2 {
      // CHECK: %[[REV_G0KITER:.+]] = affine.apply #[[REV_MAP_G0K]](%[[G0KITER]])
  rock.gridwise_attention_accel(%0, %arg1, %arg2, %arg3) features =  mfma|dot|atomic_add|atomic_add_f16 preSoftmaxOps = {} {
    arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-",
    blockSize = 64 : i32,
    gridSize = 24 : i32,
    params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>,
    params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>,
    firstGemmIdx = 0 : i32,
    operand_segment_sizes = array<i32: 1, 1, 1, 0, 0, 1>
  } : memref<1x64x384xf32>, memref<1x64x384xf32>, memref<1x384x64xf32>, memref<1x384x64xf32>
  return
}

// -----

// CHECK: @gridwise_attn_issue_1661_workaround
func.func @gridwise_attn_issue_1661_workaround(%arg0: memref<256xf16>, %arg1: memref<98304xf16>, %arg2: memref<98304xf16>, %arg3: memref<256xf16>) attributes {block_size = 32 : i32, grid_size = 4 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 + d1) * 64 + d2)> by [<Unmerge{4, 1, 64} ["g", "seq_q", "head_qk"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [4, 1, 64] -> [256]> : memref<256xf16> to memref<4x1x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 384 + d2)> by [<Unmerge{4, 64, 384} ["g", "seq_k", "head_qk"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [4, 64, 384] -> [98304]> : memref<98304xf16> to memref<4x64x384xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 384 + d1) * 64 + d2)> by [<Unmerge{4, 384, 64} ["g", "seq_k", "head_v"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [4, 384, 64] -> [98304]> : memref<98304xf16> to memref<4x384x64xf16>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> ((d0 + d1) * 64 + d2)> by [<Unmerge{4, 1, 64} ["g", "seq_q", "head_v"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [4, 1, 64] -> [256]> : memref<256xf16> to memref<4x1x64xf16>
  %4 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [4, 64, 1] -> [4, 1, 64]> : memref<4x1x64xf16> to memref<4x64x1xf16>
  %5 = rock.transform %4 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K"] at [1] -> ["gemm0K"] at [1]>, <Pad{0, 31} ["gemm0NPad"] at [2] -> ["gemm0N"] at [2]>] bounds = [4, 64, 32] -> [4, 64, 1]> : memref<4x64x1xf16> to memref<4x64x32xf16>
  %6 = rock.transform %3 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 31} ["gemm1NPad"] at [1] -> ["gemm1N"] at [1]>, <PassThrough ["gemm1M"] at [2] -> ["gemm1M"] at [2]>] bounds = [4, 32, 64] -> [4, 1, 64]> : memref<4x1x64xf16> to memref<4x32x64xf16>

  // CHECK: %[[neginf:.+]] = arith.constant 0xFC00 : f16
  // CHECK: rock.transforming_for {useIndexDiffs}
  // CHECK: %[[cmpres:.*]] = arith.cmpi eq, %{{.*}}, %false : i1
  // CHECK-NEXT: scf.if %[[cmpres]]
  // CHECK-NEXT: rock.in_bounds_store %[[neginf]] -> %{{.*}}[%{{.*}}] : f16 -> memref<32xf16, #gpu.address_space<private>>, index
  rock.gridwise_attention_accel(%5, %1, %2, %6) features =  dot|atomic_add|atomic_fmax_f32|wmma preSoftmaxOps = {
  } {arch = "amdgcn-amd-amdhsa:gfx1100", blockSize = 32 : i32, firstGemmIdx = 0 : i32, gridSize = 4 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1>, params0 = #rock.wmma_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, params1 = #rock.wmma_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, prePadG0N = 1 : index} : memref<4x64x32xf16>, memref<4x64x384xf16>, memref<4x384x64xf16>, memref<4x32x64xf16>
  return
}

// -----

// CHECK: @gridwise_attn_kvcache
func.func @gridwise_attn_kvcache(%arg0: memref<1x384x64xf32>, %arg1: memref<1x64x384xf32>, %arg2: memref<1x384x64xf32>, %arg3: memref<1x384x64xf32>, %arg4: memref<1xi32>) attributes {block_size = 64 : i32, grid_size = 24 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>
  // CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[c31:.+]] = arith.constant 31 : index
  // CHECK-DAG: %[[c32:.+]] = arith.constant 32 : index
  // CHECK: %[[currSeqLenTensor:.+]] = rock.transform %arg4 by #{{.+}} : memref<1xi32> to memref<1x1xi32>
  // CHECK: %[[registers:.+]] = rock.alloc() : memref<1xi32, #gpu.address_space<private>>
  // CHECK-NEXT: rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%[[currSeqLenTensor]]) [%{{.+}}] -> %[[registers]] : memref<1x1xi32> -> memref<1xi32, #gpu.address_space<private>>, vector<1xi1>
  // CHECK-NEXT: %[[currSeqLen:.+]] = rock.in_bounds_load %[[registers]][%[[c0]]] : memref<1xi32, #gpu.address_space<private>>, index -> i32
  // CHECK-NEXT: %[[currSeqLenIndex:.+]] = arith.index_cast %[[currSeqLen]] : i32 to index
  // CHECK: %[[num:.+]] = arith.addi %[[currSeqLenIndex]], %[[c31]] : index
  // CHECK-NEXT: %[[numIter:.+]] = arith.divui %[[num]], %[[c32]] : index
  // CHECK-NEXT: %[[lastIter:.+]] = arith.subi %[[numIter]], %[[c1]] : index
  // CHECK-NEXT: scf.for %[[iterIndex:.+]] = %[[c0]] to %[[numIter]] step %[[c1]] {
  // CHECK: %[[comparison:.+]] = arith.cmpi eq, %[[iterIndex]], %[[lastIter]] : index
  // CHECK-NEXT: scf.if %[[comparison]] {
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs} (%[[dim0:.+]], %[[dim1:.+]], %[[dim2:.+]]) = [{{.*}}]({{.*}}), ({{.*}}) = []
  // CHECK-NEXT: %[[secondComparison:.+]] = arith.cmpi uge, %[[dim2]], %[[currSeqLenIndex]] : index
  // CHECK-NEXT: scf.if %[[secondComparison]] {
  // CHECK-NEXT: rock.in_bounds_store
  rock.gridwise_attention_accel(%0, %arg1, %arg2, %arg4, %arg3) features =  mfma|dot|atomic_add|atomic_add_f16 preSoftmaxOps = {} {
    arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-",
    blockSize = 64 : i32,
    gridSize = 24 : i32,
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 1>,
    params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>,
    params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>,
    firstGemmIdx = 0 : i32
  } : memref<1x64x384xf32>, memref<1x64x384xf32>, memref<1x384x64xf32>, memref<1xi32>, memref<1x384x64xf32>
  return
}
