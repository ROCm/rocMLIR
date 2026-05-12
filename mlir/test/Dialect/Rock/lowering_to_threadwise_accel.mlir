// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -rock-blockwise-load-tile-to-threadwise -rock-blockwise-gemm-to-threadwise -canonicalize --mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: @rock_gemm_schedulev2
func.func @rock_gemm_schedulev2(%arg0: memref<1x128x128xf16>, %arg1: memref<1x128x115200xf16>, %arg2: memref<1x128x115200xf32>) attributes {block_size = 256 : i32, rock.enable_splitk_for_tuning, grid_size = 3600 : i32, rock.kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942", numCU = 228 : i32} {
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
    // CHECK: scf.for 
    // CHECK-SAME: %[[c0]] to %[[c2]] step %[[c1]]
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<16xf16, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into
    // CHECK-SAME: memref<16xf16, #gpu.address_space<private>>
    // CHECK: name = "GlobalRead"
    // CHECK: rock.stage 
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all 
    // CHECK-SAME: memref<256x16xvector<8xf16>, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all 
    // CHECK-SAME: memref<256x16xvector<8xf16>, #gpu.address_space<workgroup>>
    // CHECK: name = "LDSWrite"
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<8xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into
    // CHECK-SAME: memref<8xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: name = "LDSRead"
    // CHECK: rock.stage
    // CHECK: affine.for 
    // CHECK-SAME: 0 to 1
    // CHECK: memref.subview
    // CHECK-SAME: memref<8xvector<4xf16>,
    // CHECK: %[[AReg:.*]] = rock.transform
    // CHECK-SAME: memref<1x8xvector<4xf16>
    // CHECK: affine.for 
    // CHECK-SAME: 0 to 1
    // CHECK: memref.subview
    // CHECK-SAME: memref<8xvector<4xf16>,
    // CHECK: %[[BReg:.*]] = rock.transform
    // CHECK-SAME: memref<1x8xvector<4xf16>
    // CHECK: affine.for
    // CHECK-SAME: 0 to 8
    // CHECK: %[[outReg:.*]] = rock.transform 
    // CHECK-SAME: memref<1x1xvector<16xf32>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_gemm_accel %[[outReg]] +=  %[[AReg]] * %[[BReg]]
    // CHECK-SAME: scheduleVersion = 2
    // CHECK: name = "MMA"
    // CHECK: pipeline = #rock.rock.pipeline<1>
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) {blockSize = 256 : i32, gridSize = 3600 : i32, params = #rock.accel_gemm_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<1x128x128xf16>, memref<1x128x115200xf16>, memref<1x128x115200xf32>
  return
}

// CHECK-LABEL: @rock_gemm_schedulev1
func.func @rock_gemm_schedulev1(%arg0: memref<1x128x128xf16>, %arg1: memref<1x128x115200xf16>, %arg2: memref<1x128x115200xf32>) attributes {block_size = 256 : i32, rock.enable_splitk_for_tuning, grid_size = 3600 : i32, rock.kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942", numCU = 228 : i32} {
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
    // CHECK: scf.for 
    // CHECK-SAME: %[[c0]] to %[[c2]] step %[[c1]]
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<16xf16, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into
    // CHECK-SAME: memref<16xf16, #gpu.address_space<private>>
    // CHECK: name = "GlobalRead"
    // CHECK: rock.stage 
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all 
    // CHECK-SAME: memref<256x16xvector<8xf16>, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all 
    // CHECK-SAME: memref<256x16xvector<8xf16>, #gpu.address_space<workgroup>>
    // CHECK: name = "LDSWrite"
    // CHECK: rock.stage
    // CHECK: affine.for 
    // CHECK-SAME: 0 to 1
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<8xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: %[[AReg:.*]] = rock.transform 
    // CHECK-SAME: memref<1x8xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: affine.for 
    // CHECK-SAME: 0 to 1
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<8xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: %[[BReg:.*]] = rock.transform 
    // CHECK-SAME: memref<1x8xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: affine.for
    // CHECK-SAME: 0 to 8
    // CHECK: %[[outReg:.*]] = rock.transform 
    // CHECK-SAME: memref<1x1xvector<16xf32>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_gemm_accel %[[outReg]] +=  %[[AReg]] * %[[BReg]]
    // CHECK-SAME: scheduleVersion = 1
    // CHECK: name = "MMA"
    // CHECK: pipeline = #rock.rock.pipeline<2>
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) {blockSize = 256 : i32, gridSize = 3600 : i32, params = #rock.accel_gemm_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<1x128x128xf16>, memref<1x128x115200xf16>, memref<1x128x115200xf32>
  return
}

// CHECK-LABEL: @rock_conv_gkc01_n01gc_ngk01_0_schedulev2
func.func @rock_conv_gkc01_n01gc_ngk01_0_schedulev2(%arg0: memref<1x32x32xf16>, %arg1: memref<1x32x25600xf16>, %arg2: memref<1x32x25600xf32>) attributes {block_size = 256 : i32, rock.enable_splitk_for_tuning, grid_size = 400 : i32, rock.kernel = 0 : i32, mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-", numCU = 304 : i32} {
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<4xf16, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into
    // CHECK-SAME: memref<8xf16, #gpu.address_space<private>>
    // CHECK: name = "GlobalRead"
    // CHECK: rock.stage 
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all 
    // CHECK-SAME: memref<256x4xvector<4xf16>, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all 
    // CHECK-SAME: memref<256x8xvector<4xf16>, #gpu.address_space<workgroup>>
    // CHECK: name = "LDSWrite"
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<4xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into
    // CHECK-SAME: memref<2xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: name = "LDSRead"
    // CHECK: rock.stage
    // CHECK: affine.for 
    // CHECK-SAME: 0 to 2
    // CHECK: memref.subview
    // CHECK-SAME: memref<2xvector<4xf16>,
    // CHECK: %[[AReg:.*]] = rock.transform
    // CHECK-SAME: memref<1x2xvector<4xf16>
    // CHECK: affine.for 
    // CHECK-SAME: 0 to 1
    // CHECK: memref.subview
    // CHECK-SAME: memref<2xvector<4xf16>,
    // CHECK: %[[BReg:.*]] = rock.transform
    // CHECK-SAME: memref<1x2xvector<4xf16>
    // CHECK: affine.for
    // CHECK-SAME: 0 to 2
    // CHECK: %[[outReg:.*]] = rock.transform 
    // CHECK-SAME: memref<2x1xvector<4xf32>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_gemm_accel %[[outReg]] +=  %[[AReg]] * %[[BReg]]
    // CHECK-SAME: scheduleVersion = 2
    // CHECK: name = "MMA"
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) {blockSize = 256 : i32, gridSize = 400 : i32, params = #rock.accel_gemm_params<kpackPerBlock = 8, mPerBlock = 32, nPerBlock = 64, kpack = 4, mPerWave = 32, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<1x32x32xf16>, memref<1x32x25600xf16>, memref<1x32x25600xf32>
  return
}


// CHECK-LABEL: @rock_conv_gkc01_n01gc_ngk01_0_schedulev1
func.func @rock_conv_gkc01_n01gc_ngk01_0_schedulev1(%arg0: memref<1x32x32xf16>, %arg1: memref<1x32x25600xf16>, %arg2: memref<1x32x25600xf32>) attributes {block_size = 256 : i32, rock.enable_splitk_for_tuning, grid_size = 400 : i32, rock.kernel = 0 : i32, mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-", numCU = 304 : i32} {
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<4xf16, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into
    // CHECK-SAME: memref<8xf16, #gpu.address_space<private>>
    // CHECK: name = "GlobalRead"
    // CHECK: rock.stage 
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all 
    // CHECK-SAME: memref<256x4xvector<4xf16>, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all 
    // CHECK-SAME: memref<256x8xvector<4xf16>, #gpu.address_space<workgroup>>
    // CHECK: name = "LDSWrite"
    // CHECK: rock.stage
    // CHECK: affine.for 
    // CHECK-SAME: 0 to 2
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<2xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: %[[AReg:.*]] = rock.transform 
    // CHECK-SAME: memref<1x2xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: affine.for 
    // CHECK-SAME: 0 to 1
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<2xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: %[[BReg:.*]] = rock.transform 
    // CHECK-SAME: memref<1x2xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: affine.for
    // CHECK-SAME: 0 to 2
    // CHECK: %[[outReg:.*]] = rock.transform 
    // CHECK-SAME: memref<2x1xvector<4xf32>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_gemm_accel %[[outReg]] +=  %[[AReg]] * %[[BReg]]
    // CHECK-SAME: scheduleVersion = 1
    // CHECK: name = "MMA"
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) {blockSize = 256 : i32, gridSize = 400 : i32, params = #rock.accel_gemm_params<kpackPerBlock = 8, mPerBlock = 32, nPerBlock = 64, kpack = 4, mPerWave = 32, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<1x32x32xf16>, memref<1x32x25600xf16>, memref<1x32x25600xf32>
  return
}

// CHECK-LABEL: @rock_scaled_gemm_transA
func.func @rock_scaled_gemm_transA(%arg0: memref<1x128x64xf4E2M1FN>, %arg1: memref<1x128x64xf4E2M1FN>, %arg2: memref<1x64x64xf32>, %arg3: memref<1x128x64xf8E8M0FNU>, %arg4: memref<1x128x64xf8E8M0FNU>) attributes {block_size = 256 : i32, rock.enable_splitk_for_tuning, grid_size = 1 : i32, rock.kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx950", rock.num_cu = 256 : i64} {
    // The hand-written scales arrive in legacy broadcasted form (K equal to
    // matrix K). The lowering compacts them to natural form
    // (G, K/kQuantBlockSize, D) so the LDS / per-thread tiles for scales are
    // 32x smaller than the data tiles.
    // CHECK-DAG: rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: memref.view {{.*}} : memref<4096xi8, #gpu.address_space<workgroup>> to memref<256xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
    // CHECK-DAG: memref.view {{.*}} : memref<256xi8, #gpu.address_space<workgroup>> to memref<256xf8E8M0FNU, #gpu.address_space<workgroup>>
    // CHECK: rock.alloc() : memref<2xvector<32xf4E2M1FN>, #gpu.address_space<private>>
    // CHECK: rock.alloc() : memref<2xvector<32xf4E2M1FN>, #gpu.address_space<private>>
    // CHECK: rock.alloc() : memref<1xvector<16xf32>, #gpu.address_space<private>>
    // CHECK: rock.alloc() : memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.alloc() : memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f8E8M0FNU> -> memref<1xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f8E8M0FNU> -> memref<1xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f4E2M1FN> -> memref<32xf4E2M1FN, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f4E2M1FN> -> memref<32xf4E2M1FN, #gpu.address_space<private>>
    // CHECK: name = "GlobalRead"
    // CHECK: rock.stage
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all {{.*}} memref<1xf8E8M0FNU, #gpu.address_space<private>> -> memref<256x1xf8E8M0FNU, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all {{.*}} memref<1xf8E8M0FNU, #gpu.address_space<private>> -> memref<256x1xf8E8M0FNU, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all {{.*}} memref<32xf4E2M1FN, #gpu.address_space<private>> -> memref<256x32xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all {{.*}} memref<32xf4E2M1FN, #gpu.address_space<private>> -> memref<256x32xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
    // CHECK: name = "LDSWrite"
    // CHECK: rock.stage
    // CHECK: affine.for
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>> -> memref<2xvector<32xf4E2M1FN>, #gpu.address_space<private>>
    // CHECK: rock.transform {{.*}} memref<2xvector<32xf4E2M1FN>, #gpu.address_space<private>> to memref<1x2xvector<32xf4E2M1FN>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xf8E8M0FNU, #gpu.address_space<workgroup>> -> memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.transform {{.*}} memref<2xf8E8M0FNU, #gpu.address_space<private>> to memref<1x2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: affine.for
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>> -> memref<2xvector<32xf4E2M1FN>, #gpu.address_space<private>>
    // CHECK: rock.transform {{.*}} memref<2xvector<32xf4E2M1FN>, #gpu.address_space<private>> to memref<1x2xvector<32xf4E2M1FN>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xf8E8M0FNU, #gpu.address_space<workgroup>> -> memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.transform {{.*}} memref<2xf8E8M0FNU, #gpu.address_space<private>> to memref<1x2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: affine.for
    // CHECK: rock.threadwise_gemm_accel {{.*}} scaled by {{.*}} * {{.*}} scaled by {{.*}} : memref<1x1xvector<16xf32>, #gpu.address_space<private>> += memref<1x2xvector<32xf4E2M1FN>, #gpu.address_space<private>> scaled by memref<1x2xf8E8M0FNU, #gpu.address_space<private>> * memref<1x2xvector<32xf4E2M1FN>, #gpu.address_space<private>> scaled by memref<1x2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: name = "MMA"
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2, %arg3, %arg4) storeMethod( set) {blockSize = 256 : i32, gridSize = 1 : i32, params = #rock.accel_gemm_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 64, kpack = 32, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<1x128x64xf4E2M1FN>, memref<1x128x64xf4E2M1FN>, memref<1x64x64xf32>, memref<1x128x64xf8E8M0FNU>, memref<1x128x64xf8E8M0FNU>
  return
}

// CHECK-LABEL: @rock_scaled_gemm_transB
func.func @rock_scaled_gemm_transB(%arg0: memref<1x128x64xf4E2M1FN>, %arg1: memref<1x128x64xf4E2M1FN>, %arg2: memref<1x64x64xf32>, %arg3: memref<1x128x64xf8E8M0FNU>, %arg4: memref<1x128x64xf8E8M0FNU>) attributes {block_size = 256 : i32, rock.enable_splitk_for_tuning, grid_size = 1 : i32, rock.kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx950", rock.num_cu = 256 : i64} {
    // Scales arrive in legacy broadcasted form and are compacted to natural
    // form by the lowering, shrinking the LDS / per-thread tiles by 32x.
    // CHECK-DAG: rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: memref.view {{.*}} : memref<4096xi8, #gpu.address_space<workgroup>> to memref<256xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
    // CHECK-DAG: memref.view {{.*}} : memref<256xi8, #gpu.address_space<workgroup>> to memref<256xf8E8M0FNU, #gpu.address_space<workgroup>>
    // CHECK: rock.alloc() : memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.alloc() : memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f8E8M0FNU> -> memref<1xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f8E8M0FNU> -> memref<1xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f4E2M1FN> -> memref<32xf4E2M1FN, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f4E2M1FN> -> memref<32xf4E2M1FN, #gpu.address_space<private>>
    // CHECK: name = "GlobalRead"
    // CHECK: rock.stage
    // CHECK: rock.threadwise_write_all {{.*}} memref<1xf8E8M0FNU, #gpu.address_space<private>> -> memref<256x1xf8E8M0FNU, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_write_all {{.*}} memref<1xf8E8M0FNU, #gpu.address_space<private>> -> memref<256x1xf8E8M0FNU, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_write_all {{.*}} memref<32xf4E2M1FN, #gpu.address_space<private>> -> memref<256x32xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_write_all {{.*}} memref<32xf4E2M1FN, #gpu.address_space<private>> -> memref<256x32xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
    // CHECK: name = "LDSWrite"
    // CHECK: rock.stage
    // CHECK: affine.for
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>> -> memref<2xvector<32xf4E2M1FN>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xf8E8M0FNU, #gpu.address_space<workgroup>> -> memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: affine.for
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>> -> memref<2xvector<32xf4E2M1FN>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xf8E8M0FNU, #gpu.address_space<workgroup>> -> memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: affine.for
    // CHECK: rock.threadwise_gemm_accel {{.*}} scaled by {{.*}} * {{.*}} scaled by {{.*}} : memref<1x1xvector<16xf32>, #gpu.address_space<private>> += memref<1x2xvector<32xf4E2M1FN>, #gpu.address_space<private>> scaled by memref<1x2xf8E8M0FNU, #gpu.address_space<private>> * memref<1x2xvector<32xf4E2M1FN>, #gpu.address_space<private>> scaled by memref<1x2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: name = "MMA"
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2, %arg3, %arg4) storeMethod( set) {blockSize = 256 : i32, gridSize = 1 : i32, params = #rock.accel_gemm_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 64, kpack = 32, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<1x128x64xf4E2M1FN>, memref<1x128x64xf4E2M1FN>, memref<1x64x64xf32>, memref<1x128x64xf8E8M0FNU>, memref<1x128x64xf8E8M0FNU>
  return
}

// CHECK-LABEL: @rock_scaled_gemm_transAB
func.func @rock_scaled_gemm_transAB(%arg0: memref<1x128x64xf4E2M1FN>, %arg1: memref<1x128x64xf4E2M1FN>, %arg2: memref<1x64x64xf32>, %arg3: memref<1x128x64xf8E8M0FNU>, %arg4: memref<1x128x64xf8E8M0FNU>) attributes {block_size = 256 : i32, rock.enable_splitk_for_tuning, grid_size = 1 : i32, rock.kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx950", rock.num_cu = 256 : i64} {
    // Scales arrive in legacy broadcasted form and are compacted to natural
    // form by the lowering, shrinking the LDS / per-thread tiles by 32x.
    // CHECK-DAG: rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: memref.view {{.*}} : memref<4096xi8, #gpu.address_space<workgroup>> to memref<256xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
    // CHECK-DAG: memref.view {{.*}} : memref<256xi8, #gpu.address_space<workgroup>> to memref<256xf8E8M0FNU, #gpu.address_space<workgroup>>
    // CHECK: rock.alloc() : memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.alloc() : memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f8E8M0FNU> -> memref<1xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f8E8M0FNU> -> memref<1xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f4E2M1FN> -> memref<32xf4E2M1FN, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f4E2M1FN> -> memref<32xf4E2M1FN, #gpu.address_space<private>>
    // CHECK: name = "GlobalRead"
    // CHECK: rock.stage
    // CHECK: rock.threadwise_write_all {{.*}} memref<1xf8E8M0FNU, #gpu.address_space<private>> -> memref<256x1xf8E8M0FNU, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_write_all {{.*}} memref<1xf8E8M0FNU, #gpu.address_space<private>> -> memref<256x1xf8E8M0FNU, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_write_all {{.*}} memref<32xf4E2M1FN, #gpu.address_space<private>> -> memref<256x32xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_write_all {{.*}} memref<32xf4E2M1FN, #gpu.address_space<private>> -> memref<256x32xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
    // CHECK: name = "LDSWrite"
    // CHECK: rock.stage
    // CHECK: affine.for
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>> -> memref<2xvector<32xf4E2M1FN>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xf8E8M0FNU, #gpu.address_space<workgroup>> -> memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: affine.for
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>> -> memref<2xvector<32xf4E2M1FN>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xf8E8M0FNU, #gpu.address_space<workgroup>> -> memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: affine.for
    // CHECK: rock.threadwise_gemm_accel {{.*}} scaled by {{.*}} * {{.*}} scaled by {{.*}} : memref<1x1xvector<16xf32>, #gpu.address_space<private>> += memref<1x2xvector<32xf4E2M1FN>, #gpu.address_space<private>> scaled by memref<1x2xf8E8M0FNU, #gpu.address_space<private>> * memref<1x2xvector<32xf4E2M1FN>, #gpu.address_space<private>> scaled by memref<1x2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: name = "MMA"
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2, %arg3, %arg4) storeMethod( set) {blockSize = 256 : i32, gridSize = 1 : i32, params = #rock.accel_gemm_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 64, kpack = 32, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<1x128x64xf4E2M1FN>, memref<1x128x64xf4E2M1FN>, memref<1x64x64xf32>, memref<1x128x64xf8E8M0FNU>, memref<1x128x64xf8E8M0FNU>
  return
}

// CHECK-LABEL: @rock_scaled_gemm_no_transpose
func.func @rock_scaled_gemm_no_transpose(%arg0: memref<1x128x64xf4E2M1FN>, %arg1: memref<1x128x64xf4E2M1FN>, %arg2: memref<1x64x64xf32>, %arg3: memref<1x128x64xf8E8M0FNU>, %arg4: memref<1x128x64xf8E8M0FNU>) attributes {block_size = 256 : i32, rock.enable_splitk_for_tuning, grid_size = 1 : i32, rock.kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx950", rock.num_cu = 256 : i64} {
    // Scales arrive in legacy broadcasted form and are compacted to natural
    // form by the lowering, shrinking the LDS / per-thread tiles by 32x.
    // CHECK-DAG: rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: memref.view {{.*}} : memref<4096xi8, #gpu.address_space<workgroup>> to memref<256xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
    // CHECK-DAG: memref.view {{.*}} : memref<256xi8, #gpu.address_space<workgroup>> to memref<256xf8E8M0FNU, #gpu.address_space<workgroup>>
    // CHECK: rock.alloc() : memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.alloc() : memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f8E8M0FNU> -> memref<1xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f8E8M0FNU> -> memref<1xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f4E2M1FN> -> memref<32xf4E2M1FN, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<{{.*}}f4E2M1FN> -> memref<32xf4E2M1FN, #gpu.address_space<private>>
    // CHECK: name = "GlobalRead"
    // CHECK: rock.stage
    // CHECK: rock.threadwise_write_all {{.*}} memref<1xf8E8M0FNU, #gpu.address_space<private>> -> memref<256x1xf8E8M0FNU, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_write_all {{.*}} memref<1xf8E8M0FNU, #gpu.address_space<private>> -> memref<256x1xf8E8M0FNU, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_write_all {{.*}} memref<32xf4E2M1FN, #gpu.address_space<private>> -> memref<256x32xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_write_all {{.*}} memref<32xf4E2M1FN, #gpu.address_space<private>> -> memref<256x32xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
    // CHECK: name = "LDSWrite"
    // CHECK: rock.stage
    // CHECK: affine.for
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>> -> memref<2xvector<32xf4E2M1FN>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xf8E8M0FNU, #gpu.address_space<workgroup>> -> memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: affine.for
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>> -> memref<2xvector<32xf4E2M1FN>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into {{.*}} memref<256x1x2xf8E8M0FNU, #gpu.address_space<workgroup>> -> memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: affine.for
    // CHECK: rock.threadwise_gemm_accel {{.*}} scaled by {{.*}} * {{.*}} scaled by {{.*}} : memref<1x1xvector<16xf32>, #gpu.address_space<private>> += memref<1x2xvector<32xf4E2M1FN>, #gpu.address_space<private>> scaled by memref<1x2xf8E8M0FNU, #gpu.address_space<private>> * memref<1x2xvector<32xf4E2M1FN>, #gpu.address_space<private>> scaled by memref<1x2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: name = "MMA"
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2, %arg3, %arg4) storeMethod( set) {blockSize = 256 : i32, gridSize = 1 : i32, params = #rock.accel_gemm_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 64, kpack = 32, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<1x128x64xf4E2M1FN>, memref<1x128x64xf4E2M1FN>, memref<1x64x64xf32>, memref<1x128x64xf8E8M0FNU>, memref<1x128x64xf8E8M0FNU>
  return
}

// Same shapes as @rock_scaled_gemm_no_transpose, but the scale operands
// arrive in *natural* form (K/32 = 4 instead of 128). The lowering must
// accept them as-is (no `compactBroadcastedScale` view chain needed) and
// produce the same compact LDS / per-thread layout.
// CHECK-LABEL: @rock_scaled_gemm_natural_form_input
func.func @rock_scaled_gemm_natural_form_input(%arg0: memref<1x128x64xf4E2M1FN>, %arg1: memref<1x128x64xf4E2M1FN>, %arg2: memref<1x64x64xf32>, %arg3: memref<1x4x64xf8E8M0FNU>, %arg4: memref<1x4x64xf8E8M0FNU>) attributes {block_size = 256 : i32, rock.enable_splitk_for_tuning, grid_size = 1 : i32, rock.kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx950", rock.num_cu = 256 : i64} {
    // Natural-form scales: same per-thread / per-block scale tile sizes
    // as the broadcasted variant above (everything past `compactBroadcastedScale`
    // is shape-identical).
    // CHECK-DAG: rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
    // CHECK-DAG: rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
    // CHECK: rock.alloc() : memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.alloc() : memref<2xf8E8M0FNU, #gpu.address_space<private>>
    // CHECK: rock.threadwise_gemm_accel {{.*}} scaled by {{.*}} * {{.*}} scaled by {{.*}} : memref<1x1xvector<16xf32>, #gpu.address_space<private>> += memref<1x2xvector<32xf4E2M1FN>, #gpu.address_space<private>> scaled by memref<1x2xf8E8M0FNU, #gpu.address_space<private>> * memref<1x2xvector<32xf4E2M1FN>, #gpu.address_space<private>> scaled by memref<1x2xf8E8M0FNU, #gpu.address_space<private>>
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2, %arg3, %arg4) storeMethod( set) {blockSize = 256 : i32, gridSize = 1 : i32, params = #rock.accel_gemm_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 64, kpack = 32, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<1x128x64xf4E2M1FN>, memref<1x128x64xf4E2M1FN>, memref<1x64x64xf32>, memref<1x4x64xf8E8M0FNU>, memref<1x4x64xf8E8M0FNU>
  return
}

// Tiny-tile case where the natural-form scale tile has fewer elements
// than the workgroup, so `useNaturalScale` flips to false and the lowering
// re-broadcasts the scale tile back along K (legacy load path) for that
// operand. Look for the `vector<32xf8E8M0FNU>` per-thread register, which
// only appears in the broadcasted-fallback path; the natural-form path
// uses scalar `f8E8M0FNU` per-thread elements.
// CHECK-LABEL: @rock_scaled_gemm_fallback_broadcast
func.func @rock_scaled_gemm_fallback_broadcast(%arg0: memref<1x128x64xf4E2M1FN>, %arg1: memref<1x128x64xf4E2M1FN>, %arg2: memref<1x64x64xf32>, %arg3: memref<1x4x64xf8E8M0FNU>, %arg4: memref<1x4x64xf8E8M0FNU>) attributes {block_size = 1024 : i32, rock.enable_splitk_for_tuning, grid_size = 1 : i32, rock.kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx950", rock.num_cu = 256 : i64} {
    // blockSize=1024 with naturalKPerBlock*dPerBlock = 4*64 = 256 < 1024
    // forces both scale operands onto the broadcasted-fallback path.
    // CHECK: vector<32xf8E8M0FNU>
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2, %arg3, %arg4) storeMethod( set) {blockSize = 1024 : i32, gridSize = 1 : i32, params = #rock.accel_gemm_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 64, kpack = 32, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<1x128x64xf4E2M1FN>, memref<1x128x64xf4E2M1FN>, memref<1x64x64xf32>, memref<1x4x64xf8E8M0FNU>, memref<1x4x64xf8E8M0FNU>
  return
}

// CHECK-LABEL: @gridwise_attn_schedulev2
func.func @gridwise_attn_schedulev2(%arg0: memref<1x384x64xf32>, %arg1: memref<1x64x384xf32>, %arg2: memref<1x384x64xf32>, %arg3: memref<1x384x64xf32>) attributes {block_size = 64 : i32, grid_size = 24 : i32, rock.kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>

  // CHECK: rock.threadwise_gemm_accel
  // CHECK-SAME: scheduleVersion = 2
  
  // CHECK: rock.threadwise_gemm_accel
  // CHECK-SAME: scheduleVersion = 2
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
