// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -rock-pipeline %s | FileCheck %s

#xdlops_gemm_params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kpack = 8, mPerWave = 64, nPerWave = 64, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>
// CHECK-LABEL: @fp8_bf8_xdlops
func.func @fp8_bf8_xdlops(%arg0: memref<1x128x128xf8E4M3FNUZ>, %arg1: memref<1x128x115200xf8E5M2FNUZ>, %arg2: memref<1x128x115200xf32>) attributes {block_size = 256 : i32, grid_size = 900 : i32} {
  // The tuning testcase leads to padded buffers, we simplify here.
  // CHECK: %[[ldsA:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ldsB:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>

  // CHECK: %[[viewAStore:.+]] = memref.view %[[ldsA]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E4M3FNUZ>, #gpu.address_space<workgroup>>
  // CHECK: %[[viewBStore:.+]] = memref.view %[[ldsB]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E5M2FNUZ>, #gpu.address_space<workgroup>>
  // CHECK: %[[viewAGemm:.+]] = memref.view %[[ldsA]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E4M3FNUZ>, #gpu.address_space<workgroup>>
  // CHECK: %[[viewBGemm:.+]] = memref.view %[[ldsB]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E5M2FNUZ>, #gpu.address_space<workgroup>>

  // CHECK: %[[viewAStoreMB:.+]] = rock.extract_multibuffer(%[[viewAStore]])
  // CHECK: %[[viewAStoreTr0:.+]] = rock.transform %[[viewAStoreMB]]
  // CHECK: %[[viewAStoreTr1:.+]] = rock.transform %[[viewAStoreTr0]]
  // CHECK: %[[viewAStoreTr2:.+]] = rock.transform %[[viewAStoreTr1]]
  // CHECK: %[[viewAStoreTr3:.+]] = rock.transform %[[viewAStoreTr2]]
  // CHECK: rock.threadwise_write_all {{.*}} -> [](%[[viewAStoreTr3]])

  // CHECK: %[[viewBStoreMB:.+]] = rock.extract_multibuffer(%[[viewBStore]])
  // CHECK: %[[viewBStoreTr0:.+]] = rock.transform %[[viewBStoreMB]]
  // CHECK: %[[viewBStoreTr1:.+]] = rock.transform %[[viewBStoreTr0]]
  // CHECK: %[[viewBStoreTr2:.+]] = rock.transform %[[viewBStoreTr1]]
  // CHECK: %[[viewBStoreTr3:.+]] = rock.transform %[[viewBStoreTr2]]


  // CHECK: rock.threadwise_write_all {{.*}} -> [](%[[viewBStoreTr3]])
  // CHECK: %[[viewAGemmMB:.+]] = rock.extract_multibuffer(%[[viewAGemm]])
  // CHECK: %[[viewBGemmMB:.+]] = rock.extract_multibuffer(%[[viewBGemm]])

  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: %[[viewAGemmMB]]
  // CHECK-SAME: %[[viewBGemmMB]]
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx940", blockSize = 256 : i32, gridSize = 900 : i32, numCU = 228 : i32, params = #xdlops_gemm_params1} : memref<1x128x128xf8E4M3FNUZ>, memref<1x128x115200xf8E5M2FNUZ>, memref<1x128x115200xf32>
  return
}

// -----

#xdlops_gemm_params1a = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kpack = 8, mPerWave = 64, nPerWave = 64, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>
// CHECK-LABEL: @fp8_bf8_xdlops_ocp
func.func @fp8_bf8_xdlops_ocp(%arg0: memref<1x128x128xf8E4M3FN>, %arg1: memref<1x128x115200xf8E5M2>, %arg2: memref<1x128x115200xf32>) attributes {block_size = 256 : i32, grid_size = 900 : i32} {
  // The tuning testcase leads to padded buffers, we simplify here.
  // CHECK: %[[ldsA:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ldsB:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>

  // CHECK: %[[viewAStore:.+]] = memref.view %[[ldsA]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E4M3FN>, #gpu.address_space<workgroup>>
  // CHECK: %[[viewBStore:.+]] = memref.view %[[ldsB]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E5M2>, #gpu.address_space<workgroup>>
  // CHECK: %[[viewAGemm:.+]] = memref.view %[[ldsA]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E4M3FN>, #gpu.address_space<workgroup>>
  // CHECK: %[[viewBGemm:.+]] = memref.view %[[ldsB]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E5M2>, #gpu.address_space<workgroup>>

  // CHECK: %[[viewAStoreMB:.+]] = rock.extract_multibuffer(%[[viewAStore]])
  // CHECK: %[[viewAStoreTr0:.+]] = rock.transform %[[viewAStoreMB]]
  // CHECK: %[[viewAStoreTr1:.+]] = rock.transform %[[viewAStoreTr0]]
  // CHECK: %[[viewAStoreTr2:.+]] = rock.transform %[[viewAStoreTr1]]
  // CHECK: %[[viewAStoreTr3:.+]] = rock.transform %[[viewAStoreTr2]]
  // CHECK: rock.threadwise_write_all {{.*}} -> [](%[[viewAStoreTr3]])

  // CHECK: %[[viewBStoreMB:.+]] = rock.extract_multibuffer(%[[viewBStore]])
  // CHECK: %[[viewBStoreTr0:.+]] = rock.transform %[[viewBStoreMB]]
  // CHECK: %[[viewBStoreTr1:.+]] = rock.transform %[[viewBStoreTr0]]
  // CHECK: %[[viewBStoreTr2:.+]] = rock.transform %[[viewBStoreTr1]]
  // CHECK: %[[viewBStoreTr3:.+]] = rock.transform %[[viewBStoreTr2]]


  // CHECK: rock.threadwise_write_all {{.*}} -> [](%[[viewBStoreTr3]])
  // CHECK: %[[viewAGemmMB:.+]] = rock.extract_multibuffer(%[[viewAGemm]])
  // CHECK: %[[viewBGemmMB:.+]] = rock.extract_multibuffer(%[[viewBGemm]])

  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: %[[viewAGemmMB]]
  // CHECK-SAME: %[[viewBGemmMB]]
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx940", blockSize = 256 : i32, gridSize = 900 : i32, numCU = 228 : i32, params = #xdlops_gemm_params1a} : memref<1x128x128xf8E4M3FN>, memref<1x128x115200xf8E5M2>, memref<1x128x115200xf32>
  return
}

// -----

#xdlops_gemm_params2 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 64, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>
// CHECK: #[[REV_MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
// CHECK: @reverse_grid
func.func @reverse_grid(%arg0: memref<1x32x128xf32>, %arg1: memref<1x32x256xf32>, %arg2: memref<1x128x256xf32>) attributes {block_size = 256 : i32, grid_size = 8 : i32, reverse_grid} {
  // CHECK: scf.for %[[KITER:.+]] = %c0 to %c8 step %c1 {
    // CHECK: %[[REV_KITER:.+]] = affine.apply #[[REV_MAP]](%[[KITER]])[%c8]
    // CHECK: rock.threadwise_read_into
    // CHECK-SAME: [%[[REV_KITER:.+]],
    // CHECK: rock.threadwise_read_into
    // CHECK-SAME: [%[[REV_KITER:.+]],
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx940", blockSize = 256 : i32, gridSize = 900 : i32, numCU = 228 : i32, params = #xdlops_gemm_params2} : memref<1x32x128xf32>, memref<1x32x256xf32>, memref<1x128x256xf32>
  return
}

// CHECK: @chiplet_grid
func.func @chiplet_grid(%arg0: memref<1x32x128xf32>, %arg1: memref<1x32x256xf32>, %arg2: memref<1x128x256xf32>) attributes {block_size = 256 : i32, grid_size = 8 : i32} {
  // CHECK: %[[BID:.+]] = rock.workgroup_id
  // CHECK-DAG: %[[CHIPLET_GRP_ID:.+]] = arith.remui %[[BID]], %c4 : index
  // CHECK-DAG: %[[CHIPLET_BID:.+]] = arith.divui %[[BID]], %c4 : index
  // CHECK-DAG: %[[CHIPLET_GRP_ID_LSHIFT:.+]] = arith.muli %[[CHIPLET_GRP_ID]], %c2 : index
  // CHECK-DAG: %[[MAYBE_NEW_BID:.+]] = arith.addi %[[CHIPLET_BID]], %[[CHIPLET_GRP_ID_LSHIFT]] : index
  // CHECK-DAG: %[[IS_TAIL_BID:.+]] = arith.cmpi sgt, %[[BID]], %c7 : index
  // CHECK-DAG: %[[NEW_BID:.+]] = arith.select %[[IS_TAIL_BID]], %[[BID]], %[[MAYBE_NEW_BID]] : index
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx942", blockSize = 256 : i32, gridSize = 900 : i32, numCU = 228 : i32, params = #xdlops_gemm_params2} : memref<1x32x128xf32>, memref<1x32x256xf32>, memref<1x128x256xf32>
  return
}
