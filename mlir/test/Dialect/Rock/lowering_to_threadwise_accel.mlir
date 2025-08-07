// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise %s | FileCheck %s

// CHECK-LABEL: @rock_gemm_schedulev2
func.func @rock_gemm_schedulev2(%arg0: memref<1x128x128xf16>, %arg1: memref<1x128x115200xf16>, %arg2: memref<1x128x115200xf32>) attributes {block_size = 256 : i32, enable_splitk_for_tuning, grid_size = 3600 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942", numCU = 228 : i32} {
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: memref.store 
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
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all 
    // CHECK-SAME: memref<256x16xvector<8xf16>, #gpu.address_space<workgroup>>
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
    // CHECK: rock.threadwise_accel_gemm %[[outReg]] +=  %[[AReg]] * %[[BReg]]
    // CHECK-SAME: scheduleVersion = 2
    // CHECK: name = "MMA"
    // CHECK: pipeline = #rock.pipeline<1>
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) {blockSize = 256 : i32, gridSize = 3600 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, forceUnroll = true>} : memref<1x128x128xf16>, memref<1x128x115200xf16>, memref<1x128x115200xf32>
  return
}

// CHECK-LABEL: @rock_gemm_schedulev1
func.func @rock_gemm_schedulev1(%arg0: memref<1x128x128xf16>, %arg1: memref<1x128x115200xf16>, %arg2: memref<1x128x115200xf32>) attributes {block_size = 256 : i32, enable_splitk_for_tuning, grid_size = 3600 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942", numCU = 228 : i32} {
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: memref.store 
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
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all 
    // CHECK-SAME: memref<256x16xvector<8xf16>, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_write_all 
    // CHECK-SAME: memref<256x16xvector<8xf16>, #gpu.address_space<workgroup>>
    // CHECK: name = "LDSWrite"
    // CHECK: rock.stage
    // CHECK: affine.for 
    // CHECK-SAME: 0 to 1
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<8xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: affine.for 
    // CHECK-SAME: 0 to 1
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<8xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: affine.for
    // CHECK-SAME: 0 to 8
    // CHECK: %[[AReg:.*]] = rock.transform 
    // CHECK-SAME: memref<1x8xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: %[[BReg:.*]] = rock.transform 
    // CHECK-SAME: memref<1x8xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: %[[outReg:.*]] = rock.transform 
    // CHECK-SAME: memref<1x1xvector<16xf32>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_accel_gemm %[[outReg]] +=  %[[AReg]] * %[[BReg]]
    // CHECK-SAME: scheduleVersion = 1
    // CHECK: name = "MMA"
    // CHECK: pipeline = #rock.pipeline<2>
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) {blockSize = 256 : i32, gridSize = 3600 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>} : memref<1x128x128xf16>, memref<1x128x115200xf16>, memref<1x128x115200xf32>
  return
}

// CHECK-LABEL: @rock_conv_gkc01_n01gc_ngk01_0_schedulev2
func.func @rock_conv_gkc01_n01gc_ngk01_0_schedulev2(%arg0: memref<1x32x32xf16>, %arg1: memref<1x32x25600xf16>, %arg2: memref<1x32x25600xf32>) attributes {block_size = 256 : i32, enable_splitk_for_tuning, grid_size = 400 : i32, kernel = 0 : i32, mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-", numCU = 304 : i32} {
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: memref.store 
    // CHECK-DAG: %[[c1_0:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c1_1:.+]] = arith.constant 1 : index
    // CHECK: scf.for 
    // CHECK-SAME: %[[c0]] to %[[c1_0]] step %[[c1_1]]
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<4xf16, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into
    // CHECK-SAME: memref<8xf16, #gpu.address_space<private>>
    // CHECK: name = "GlobalRead"
    // CHECK: rock.stage 
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all 
    // CHECK-SAME: memref<256x4xvector<4xf16>, #gpu.address_space<workgroup>>
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
    // CHECK: rock.threadwise_accel_gemm %[[outReg]] +=  %[[AReg]] * %[[BReg]]
    // CHECK-SAME: scheduleVersion = 2
    // CHECK: name = "MMA"
    // CHECK: pipeline = #rock.pipeline<1>
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) {blockSize = 256 : i32, gridSize = 400 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 32, nPerBlock = 64, kpack = 4, mPerWave = 32, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, forceUnroll = true>} : memref<1x32x32xf16>, memref<1x32x25600xf16>, memref<1x32x25600xf32>
  return
}


// CHECK-LABEL: @rock_conv_gkc01_n01gc_ngk01_0_schedulev1
func.func @rock_conv_gkc01_n01gc_ngk01_0_schedulev1(%arg0: memref<1x32x32xf16>, %arg1: memref<1x32x25600xf16>, %arg2: memref<1x32x25600xf32>) attributes {block_size = 256 : i32, enable_splitk_for_tuning, grid_size = 400 : i32, kernel = 0 : i32, mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-", numCU = 304 : i32} {
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: memref.store 
    // CHECK-DAG: %[[c1_0:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c1_1:.+]] = arith.constant 1 : index
    // CHECK: scf.for 
    // CHECK-SAME: %[[c0]] to %[[c1_0]] step %[[c1_1]]
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<4xf16, #gpu.address_space<private>>
    // CHECK: rock.threadwise_read_into
    // CHECK-SAME: memref<8xf16, #gpu.address_space<private>>
    // CHECK: name = "GlobalRead"
    // CHECK: rock.stage 
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_copy
    // CHECK: rock.threadwise_write_all 
    // CHECK-SAME: memref<256x4xvector<4xf16>, #gpu.address_space<workgroup>>
    // CHECK: rock.threadwise_write_all 
    // CHECK-SAME: memref<256x8xvector<4xf16>, #gpu.address_space<workgroup>>
    // CHECK: name = "LDSWrite"
    // CHECK: rock.stage
    // CHECK: affine.for 
    // CHECK-SAME: 0 to 2
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<2xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: affine.for 
    // CHECK-SAME: 0 to 1
    // CHECK: rock.threadwise_read_into 
    // CHECK-SAME: memref<2xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: affine.for
    // CHECK-SAME: 0 to 2
    // CHECK: %[[AReg:.*]] = rock.transform 
    // CHECK-SAME: memref<1x2xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: %[[BReg:.*]] = rock.transform 
    // CHECK-SAME: memref<1x2xvector<4xf16>, #gpu.address_space<private>>
    // CHECK: %[[outReg:.*]] = rock.transform 
    // CHECK-SAME: memref<2x1xvector<4xf32>, #gpu.address_space<private>>
    // CHECK: rock.threadwise_accel_gemm %[[outReg]] +=  %[[AReg]] * %[[BReg]]
    // CHECK-SAME: scheduleVersion = 1
    // CHECK: name = "MMA"
    // CHECK: pipeline = #rock.pipeline<2>
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) {blockSize = 256 : i32, gridSize = 400 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 32, nPerBlock = 64, kpack = 4, mPerWave = 32, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>} : memref<1x32x32xf16>, memref<1x32x25600xf16>, memref<1x32x25600xf32>
  return
}

