// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -canonicalize %s | FileCheck %s

#xdlops_gemm_params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 32, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor=1, forceUnroll = true>
// CHECK-LABEL: @gridwise_attn_simple
// CHECK-SAME: (%[[Q:.+]]: memref<1x384x64xf32>, %[[K:.+]]: memref<1x64x384xf32>, %[[V:.+]]: memref<1x384x64xf32>, %[[O:.+]]: memref<1x384x64xf32>)
// CHECK-DAG: %[[ln2Recip:.+]] = arith.constant 1.44269502 : f32
// CHECK-DAG: %[[negInf:.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG: %[[zeroF32:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[zeroVecF32:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>

// CHECK: %[[QTr0:.+]] = rock.transform %[[Q]] by
// CHECK: %[[ldsG0A:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
// CHECK: %[[ldsReductionWSBytes:.+]] = memref.subview {{.*}} : memref<4096xi8, #gpu.address_space<workgroup>> to memref<256xi8, #gpu.address_space<workgroup>>
// CHECK: %[[ldsG0B:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
// CHECK: %[[ldsReductionWS:.+]] = memref.view %[[ldsReductionWSBytes]][

// init maxRow buffer
// CHECK-DAG: rock.fill(%[[maxRowBuf:.+]], %[[negInf]])

// init sumRow buffer
// CHECK-DAG: rock.fill(%[[sumRowBuf:.+]], %[[zeroF32]])

// init attentionAcc buffer
// CHECK-DAG: rock.fill(%[[attnOutBuf:.+]], %[[zeroF32]])

// Outer N-tile loop
// CHECK: affine.for
  // CHECK-DAG: rock.fill(%[[gemm0AccBuf:.+]], %[[zeroVecF32]])
  // CHECK-DAG: rock.fill(%[[gemm1AccBuf:.+]], %[[zeroVecF32]])

  // Inner gemm0 KpacksPerBlock loop
  // CHECK: affine.for

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

    // Load G0B from LDS to regs and accel gemm
    // CHECK-DAG: %[[view2G0BStoreTr0:.+]] = rock.transform %[[view2G0BStore]]
    // CHECK-DAG: %[[view2G0BStoreTr1:.+]] = rock.transform %[[view2G0BStoreTr0]]
    // CHECK-DAG: %[[view2G0BStoreTr2:.+]] = rock.transform %[[view2G0BStoreTr1]]
    // CHECK-DAG: %[[view2G0BStoreTr3:.+]] = rock.transform %[[view2G0BStoreTr2]]
    // CHECK: affine.for
      // CHECK: affine.for
        // CHECK: rock.threadwise_read_into {{.*}} [](%[[view2G0BStoreTr3]]) {{.*}} -> %[[preAccelRegB:.+]] :
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
  // CHECK: rock.blockwise_broadcast_reduce max {{.*}} %[[gemm0AccBufScalar]] into %[[gemm0Max:[0-9]+]] using %[[ldsReductionWS]]

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

  // CHECK: rock.blockwise_broadcast_reduce sum {{.*}} %[[gemm0NormExp]] into %[[gemm0NormExpSum:[0-9]+]] using %[[ldsReductionWS]]

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

  // Viewing another set of register with kPack packing
  // CHECK: %[[G1AregsKpackTr0:.+]] = rock.transform %[[G1AregsKpack:.+]] by
  // CHECK-DAG: %[[G1AregsKpackTr1:.+]] = rock.transform %[[G1AregsKpackTr0]] by
  // CHECK-DAG: %[[G1AregsKpackTr2:.+]] = rock.transform %[[G1AregsKpackTr1]] by

  // CHECK-DAG: rock.threadwise_copy %[[gemm0NormExpTr2]] -> %[[G1AregsKpackTr2]]

  // Viewing G1 LDS A tile buffer
  // CHECK-DAG: %[[viewG1AStore:.+]] = memref.view %[[ldsG0A]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[viewG1AStoreTr0:.+]] = rock.transform %[[viewG1AStore]]
  // CHECK-DAG: %[[viewG1AStoreTr1:.+]] = rock.transform %[[viewG1AStoreTr0]]
  // CHECK-DAG: %[[viewG1AStoreTr2:.+]] = rock.transform %[[viewG1AStoreTr1]]
  // CHECK-DAG: %[[viewG1AStoreTr3:.+]] = rock.transform %[[viewG1AStoreTr2]]
  // CHECK-DAG: %[[viewG1AStoreTr4:.+]] = rock.transform %[[viewG1AStoreTr3]]
  // CHECK-DAG: %[[viewG1AStoreTr5:.+]] = rock.transform %[[viewG1AStoreTr4]]
  // CHECK-DAG: %[[viewG1AStoreTr6:.+]] = rock.transform %[[viewG1AStoreTr5]]

  // Store to LDS G1A tile buffer
  // CHECK-DAG: rock.threadwise_write_all {{.*}} %[[G1AregsKpack]] -> [](%[[viewG1AStoreTr6]])
  // CHECK-DAG: %[[view2G1AStore:.+]] = memref.view %[[ldsG0A]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
  
  // Viewing LDS G1A tile buffer in MFMA layout
  // CHECK-DAG: %[[viewG1ALoadTr0:.+]] = rock.transform %[[view2G1AStore]]
  // CHECK-DAG: %[[viewG1ALoadTr1:.+]] = rock.transform %[[viewG1ALoadTr0]]
  // CHECK-DAG: %[[viewG1ALoadTr2:.+]] = rock.transform %[[viewG1ALoadTr1]]
  // CHECK-DAG: %[[viewG1ALoadTr3:.+]] = rock.transform %[[viewG1ALoadTr2]]

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
  // CHECK-DAG: %[[viewG1BStore:.+]] = memref.view %[[ldsG0B]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[viewG1BStoreTr0:.+]] = rock.transform %[[viewG1BStore]]
  // CHECK-DAG: %[[viewG1BStoreTr1:.+]] = rock.transform %[[viewG1BStoreTr0]]
  // CHECK-DAG: %[[viewG1BStoreTr2:.+]] = rock.transform %[[viewG1BStoreTr1]]
  // CHECK-DAG: %[[viewG1BStoreTr3:.+]] = rock.transform %[[viewG1BStoreTr2]]

  // Store to LDS G1B tile buffer
  // CHECK-DAG: rock.threadwise_write_all {{.*}} %[[G1BregsKpack]] -> [](%[[viewG1BStoreTr3]])
  // CHECK-DAG: %[[view2G1BStore:.+]] = memref.view %[[ldsG0B]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>

  // Viewing LDS G1B tile buffer in MFMA layout
  // CHECK-DAG: %[[viewG1BLoadTr0:.+]] = rock.transform %[[view2G1BStore]]
  // CHECK-DAG: %[[viewG1BLoadTr1:.+]] = rock.transform %[[viewG1BLoadTr0]]
  // CHECK-DAG: %[[viewG1BLoadTr2:.+]] = rock.transform %[[viewG1BLoadTr1]]
  // CHECK-DAG: %[[viewG1BLoadTr3:.+]] = rock.transform %[[viewG1BLoadTr2]]

  // Gemm1
  // CHECK-DAG: rock.lds_barrier
  // CHECK: affine.for
      // CHECK: affine.for
        // CHECK: rock.threadwise_read_into {{.*}} [](%[[viewG1BLoadTr3]]) {{.*}} -> %[[preAccelRegB:.+]] :
        // CHECK: rock.threadwise_read_into {{.*}} [](%[[viewG1ALoadTr3]]) {{.*}} -> %[[preAccelRegA:.+]] :
        // CHECK: %[[bufferA:.*]] = rock.transform %[[preAccelRegB]]
        // CHECK: %[[bufferB:.*]] = rock.transform %[[preAccelRegA]]
        // CHECK: %[[bufferC:.*]] = rock.transform %[[gemm1AccBuf]]
        // CHECK: rock.threadwise_accel_gemm %[[bufferC]]{{.*}} += %[[bufferA:.*]] * %[[bufferB:.*]]

  // CHECK: rock.transforming_for
    // CHECK: %[[tmp1:.+]] =  memref.load %[[gemm1AccBuf]][
    // CHECK: rock.in_bounds_store %[[tmp1]] -> %[[gemm1AccBufScalar:.+]][

  // Reduction corrections
  // CHECK: rock.transforming_for
    // CHECK-DAG: %[[maxdiffexp:.+]] = rock.in_bounds_load %[[maxdiffexpbuf]]
    // CHECK-DAG: %[[attnOutVal:.+]] = rock.in_bounds_load %[[attnOutBuf]]
    // CHECK-DAG: %[[gemm1Val:.+]] = rock.in_bounds_load %[[gemm1AccBufScalar]]

    // CHECK-DAG: %[[attnOutBufMul:.+]] = arith.mulf %[[attnOutVal]], %[[maxdiffexp]]
    // CHECK-DAG: %[[newattnOutVal:.+]] = arith.addf %[[attnOutBufMul]], %[[gemm1Val]]
    // CHECK-DAG: rock.in_bounds_store %[[newattnOutVal]] -> %[[attnOutBuf]]
  // CHECK : }
// CHECK : }
// CHECK : rock.threadwise_write_all {{.*}} %[[attnOutBuf]] -> {{.*}}(%[[O]])

func.func @gridwise_attn_simple(%arg0: memref<1x384x64xf32>, %arg1: memref<1x64x384xf32>, %arg2: memref<1x384x64xf32>, %arg3: memref<1x384x64xf32>) attributes {block_size = 64 : i32, grid_size = 24 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>
  rock.gridwise_attention_accel(%0, %arg1, %arg2, %arg3) features =  mfma|dot|atomic_add preSoftmaxOps = {} {
    arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-",
    blockSize = 64 : i32,
    gridSize = 24 : i32,
    params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>,
    params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>,
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
  rock.gridwise_attention_accel(%0, %arg1, %arg2, %arg3) features =  mfma|dot|atomic_add preSoftmaxOps = {} {
    arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-",
    blockSize = 64 : i32,
    gridSize = 24 : i32,
    params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>,
    params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>,
    operand_segment_sizes = array<i32: 1, 1, 1, 0, 0, 1>
  } : memref<1x64x384xf32>, memref<1x64x384xf32>, memref<1x384x64xf32>, memref<1x384x64xf32>
  return
}

