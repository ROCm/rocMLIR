// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise %s | FileCheck %s

#xdlops_gemm_params = #rock.xdlops_gemm_params<kPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kpack = 8, mPerWave = 64, nPerWave = 64, forceUnroll = true>
// CHECK-LABEL: @fp8_bf8_xdlops
func.func @fp8_bf8_xdlops(%arg0: memref<1x128x128xf8E4M3FNUZ>, %arg1: memref<1x128x115200xf8E5M2FNUZ>, %arg2: memref<1x128x115200xf32>) attributes {block_size = 256 : i32, grid_size = 900 : i32} {
  // The tuning testcase leads to padded buffers, we simplify here.
  // CHECK: %[[ldsA:.+]] = rock.alloc() : memref<8192xf8E4M3FNUZ, #gpu.address_space<workgroup>>
  // CHECK: %[[ldsB:.+]] = rock.alloc() : memref<8192xf8E5M2FNUZ, #gpu.address_space<workgroup>>
  // CHECK: rock.blockwise_gemm_v2
  // CHECK-SAME %[[ldsA]]
  // CHECK-SAME: %[[ldsB]]
  rock.gridwise_gemm_v2(%arg0, %arg1, %arg2) storeMethod( set) features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx940", blockSize = 256 : i32, gridSize = 900 : i32, params = #xdlops_gemm_params} : memref<1x128x128xf8E4M3FNUZ>, memref<1x128x115200xf8E5M2FNUZ>, memref<1x128x115200xf32>
  return
}

// -----
