// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -rock-pipeline %s | FileCheck %s

#xdlops_gemm_params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kpack = 8, mPerWave = 64, nPerWave = 64, mnPerXdl = 32, forceUnroll = true>
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
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx940", blockSize = 256 : i32, gridSize = 900 : i32, numCU = 228 : i32, params = #xdlops_gemm_params} : memref<1x128x128xf8E4M3FNUZ>, memref<1x128x115200xf8E5M2FNUZ>, memref<1x128x115200xf32>
  return
}

// -----
