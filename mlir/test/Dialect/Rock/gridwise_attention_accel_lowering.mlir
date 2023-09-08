// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -canonicalize %s | FileCheck %s

#xdlops_gemm_params = #rock.xdlops_gemm_params<kpackPerBlock = 8, mPerBlock = 32, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 32, forceUnroll = true>
// CHECK-LABEL: @gridwise_attn_simple
// CHECK-SAME: (%[[Q:.+]]: memref<1x384x64xf32>, %[[K:.+]]: memref<1x64x384xf32>, %[[V:.+]]: memref<1x384x64xf32>, %[[O:.+]]: memref<1x384x64xf32>)
module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK-DAG: %[[negInf:.+]] = arith.constant 0xFF800000 : f32
  // CHECK-DAG: %[[zeroF32:.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: %[[zeroVecF32:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>

  // CHECK: %[[QTr0:.+]] = rock.transform %[[Q]] by
  // CHECK: %[[ldsG0A:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ldsG0B:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ldsG1A:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ldsReductionWS:.+]] = memref.view %[[ldsG1A]][
  // CHECK: %[[ldsG1B:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>

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

      // CHECK-DAG: %[[QTr1:.+]] = rock.transform %[[QTr0]] by
      // CHECK-DAG: %[[QTr2:.+]] = rock.transform %[[QTr1]] by
      // CHECK-DAG: rock.threadwise_read_into {{.*}}(%[[QTr2]]) {{.*}} -> %[[G0Aregs:.+]] :
      // CHECK-DAG: %[[G0AregsTr0:.+]] = rock.transform %[[G0Aregs]] by
      // CHECK-DAG: %[[G0AregsTr1:.+]] = rock.transform %[[G0AregsTr0]] by
      // CHECK: %[[G0AregsKpackTr0:.+]] = rock.transform %[[G0AregsKpack:.+]] by
      // CHECK-DAG: %[[G0AregsKpackTr1:.+]] = rock.transform %[[G0AregsKpackTr0:.+]] by
      // CHECK-DAG: rock.threadwise_transpose {{.*}} %[[G0AregsTr1]] -> %[[G0AregsKpackTr1]]

      // CHECK-DAG: %[[viewG0AStore:.+]] = memref.view %[[ldsG0A]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
      // CHECK-DAG: %[[viewG0AStoreTr0:.+]] = rock.transform %[[viewG0AStore]]
      // CHECK-DAG: %[[viewG0AStoreTr1:.+]] = rock.transform %[[viewG0AStoreTr0]]
      // CHECK-DAG: %[[viewG0AStoreTr2:.+]] = rock.transform %[[viewG0AStoreTr1]]
      // CHECK-DAG: %[[viewG0AStoreTr3:.+]] = rock.transform %[[viewG0AStoreTr2]]

      // CHECK-DAG: rock.threadwise_write_all {{.*}} %[[G0AregsKpack]] -> [](%[[viewG0AStoreTr3]])
      // CHECK-DAG: %[[view2G0AStore:.+]] = memref.view %[[ldsG0A]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>

      // CHECK-DAG: %[[KTr0:.+]] = rock.transform %[[K]] by
      // CHECK-DAG: %[[KTr1:.+]] = rock.transform %[[KTr0]] by
      // CHECK-DAG: rock.threadwise_read_into {{.*}}(%[[KTr1]]) {{.*}} -> %[[G0Bregs:.+]] :
      // CHECK-DAG: %[[G0BregsTr0:.+]] = rock.transform %[[G0Bregs]] by
      // CHECK-DAG: %[[G0BregsTr1:.+]] = rock.transform %[[G0BregsTr0]] by
      // CHECK: %[[G0BregsKpackTr0:.+]] = rock.transform %[[G0BregsKpack:.+]] by
      // CHECK-DAG: %[[G0BregsKpackTr1:.+]] = rock.transform %[[G0BregsKpackTr0:.+]] by
      // CHECK-DAG: rock.threadwise_transpose {{.*}} %[[G0BregsTr1]] -> %[[G0BregsKpackTr1]]

      // CHECK-DAG: %[[viewG0BStore:.+]] = memref.view %[[ldsG0B]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
      // CHECK-DAG: %[[viewG0BStoreTr0:.+]] = rock.transform %[[viewG0BStore]]
      // CHECK-DAG: %[[viewG0BStoreTr1:.+]] = rock.transform %[[viewG0BStoreTr0]]
      // CHECK-DAG: %[[viewG0BStoreTr2:.+]] = rock.transform %[[viewG0BStoreTr1]]
      // CHECK-DAG: %[[viewG0BStoreTr3:.+]] = rock.transform %[[viewG0BStoreTr2]]

      // CHECK-DAG: rock.threadwise_write_all {{.*}} %[[G0BregsKpack]] -> [](%[[viewG0BStoreTr3]])
      // CHECK-DAG: %[[view2G0BStore:.+]] = memref.view %[[ldsG0B]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>

      // CHECK-DAG: rock.blockwise_gemm_accel %[[gemm0AccBuf]] += {{.*}} from %[[view2G0AStore]]{{.*}} * {{.*}} from %[[view2G0BStore]]{{.*}}
    // End of inner gemm0 KpacksPerBlock loop
    // CHECK: }
    // CHECK: rock.transforming_for
      // CHECK: %[[tmp:.+]] =  memref.load %[[gemm0AccBuf]][
      // CHECK: rock.in_bounds_store %[[tmp]] -> %[[gemm0AccBufScalar:.+]][
    // CHECK: rock.blockwise_broadcast_reduce max {{.*}} %[[gemm0AccBufScalar]] into %[[gemm0Max:.+]], {{.*}} %[[gemm0MaxG1Layout:.+]] using %[[ldsReductionWS]]
    // CHECK: linalg.elemwise_binary {{.*}} fun = #linalg.binary_fn<sub>}
    // TODO(complete this test)
 
  func.func @gridwise_attn_simple(%arg0: memref<1x384x64xf32>, %arg1: memref<1x64x384xf32>, %arg2: memref<1x384x64xf32>, %arg3: memref<1x384x64xf32>) attributes {block_size = 64 : i32, grid_size = 12 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
    %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>
    rock.gridwise_attention_accel(%0, %arg1, %arg2, %arg3) features =  mfma|dot|atomic_add {
      arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-", 
      blockSize = 64 : i32, 
      gridSize = 12 : i32, 
      params = #rock.xdlops_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, forceUnroll = true>
    } : memref<1x64x384xf32>, memref<1x64x384xf32>, memref<1x384x64xf32>, memref<1x384x64xf32>
    return
  }
}

// -----
