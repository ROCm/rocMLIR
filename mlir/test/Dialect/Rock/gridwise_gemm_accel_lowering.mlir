// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -rock-blockwise-load-tile-to-threadwise -rock-pipeline %s | FileCheck %s

#xdlops_gemm_params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kpack = 8, mPerWave = 64, nPerWave = 64, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>
// CHECK-LABEL: @fp8_bf8_xdlops
func.func @fp8_bf8_xdlops(%arg0: memref<1x128x128xf8E4M3FNUZ>, %arg1: memref<1x128x115200xf8E5M2FNUZ>, %arg2: memref<1x128x115200xf32>) attributes {block_size = 256 : i32, grid_size = 900 : i32, arch = "amdgcn-amd-amdhsa:gfx942", numCU = 228 : i32} {
  // The tuning testcase leads to padded buffers, we simplify here.
  // CHECK: %[[ldsA:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ldsB:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>

  // CHECK-DAG: %[[viewAGemm:.+]] = memref.view %[[ldsA]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E4M3FNUZ>, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[viewBGemm:.+]] = memref.view %[[ldsB]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E5M2FNUZ>, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[viewAStore:.+]] = memref.view %[[ldsA]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E4M3FNUZ>, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[viewBStore:.+]] = memref.view %[[ldsB]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E5M2FNUZ>, #gpu.address_space<workgroup>>

  // CHECK-DAG: %[[viewAStore2:.+]] = memref.view %[[ldsA]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E4M3FNUZ>, #gpu.address_space<workgroup>>
  // CHECK: %[[viewAStoreMB:.+]] = rock.extract_multibuffer(%[[viewAStore2]])
  // CHECK: %[[viewAStoreTr0:.+]] = rock.transform %[[viewAStoreMB]]
  // CHECK: %[[viewAStoreTr1:.+]] = rock.transform %[[viewAStoreTr0]]
  // CHECK: %[[viewAStoreTr2:.+]] = rock.transform %[[viewAStoreTr1]]
  // CHECK: %[[viewAStoreTr3:.+]] = rock.transform %[[viewAStoreTr2]]
  // CHECK: rock.threadwise_write_all {{.*}} -> [](%[[viewAStoreTr3]])
  
  // CHECK-DAG: %[[viewBStore2:.+]] = memref.view %[[ldsB]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E5M2FNUZ>, #gpu.address_space<workgroup>>
  // CHECK: %[[viewBStoreMB:.+]] = rock.extract_multibuffer(%[[viewBStore2]])
  // CHECK: %[[viewBStoreTr0:.+]] = rock.transform %[[viewBStoreMB]]
  // CHECK: %[[viewBStoreTr1:.+]] = rock.transform %[[viewBStoreTr0]]
  // CHECK: %[[viewBStoreTr2:.+]] = rock.transform %[[viewBStoreTr1]]
  // CHECK: %[[viewBStoreTr3:.+]] = rock.transform %[[viewBStoreTr2]]


  // CHECK: rock.threadwise_write_all {{.*}} -> [](%[[viewBStoreTr3]])
  // CHECK: %[[viewAGemmMB:.+]] = rock.extract_multibuffer(%[[viewAGemm]])
  // CHECK: %[[viewBGemmMB:.+]] = rock.extract_multibuffer(%[[viewBGemm]])

  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: from %[[viewAGemmMB]]
  // CHECK-SAME: from %[[viewBGemmMB]]
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) {blockSize = 256 : i32, gridSize = 900 : i32, params = #xdlops_gemm_params1} : memref<1x128x128xf8E4M3FNUZ>, memref<1x128x115200xf8E5M2FNUZ>, memref<1x128x115200xf32>
  return
}

// -----

#xdlops_gemm_params1a = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kpack = 8, mPerWave = 64, nPerWave = 64, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>
// CHECK-LABEL: @fp8_bf8_xdlops_ocp
func.func @fp8_bf8_xdlops_ocp(%arg0: memref<1x128x128xf8E4M3FN>, %arg1: memref<1x128x115200xf8E5M2>, %arg2: memref<1x128x115200xf32>) attributes {block_size = 256 : i32, grid_size = 900 : i32, arch = "amdgcn-amd-amdhsa:gfx950", numCU = 256 : i32} {
  // The tuning testcase leads to padded buffers, we simplify here.
  // CHECK: %[[ldsA:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ldsB:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>

  // CHECK-DAG: %[[viewAGemm:.+]] = memref.view %[[ldsA]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E4M3FN>, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[viewBGemm:.+]] = memref.view %[[ldsB]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E5M2>, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[viewAStore:.+]] = memref.view %[[ldsA]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E4M3FN>, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[viewBStore:.+]] = memref.view %[[ldsB]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E5M2>, #gpu.address_space<workgroup>>

  // CHECK-DAG: %[[viewAStore2:.+]] = memref.view %[[ldsA]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E4M3FN>, #gpu.address_space<workgroup>>
  // CHECK: %[[viewAStoreMB:.+]] = rock.extract_multibuffer(%[[viewAStore2]])
  // CHECK: %[[viewAStoreTr0:.+]] = rock.transform %[[viewAStoreMB]]
  // CHECK: %[[viewAStoreTr1:.+]] = rock.transform %[[viewAStoreTr0]]
  // CHECK: %[[viewAStoreTr2:.+]] = rock.transform %[[viewAStoreTr1]]
  // CHECK: %[[viewAStoreTr3:.+]] = rock.transform %[[viewAStoreTr2]]
  // CHECK: rock.threadwise_write_all {{.*}} -> [](%[[viewAStoreTr3]])

  // CHECK-DAG: %[[viewBStore2:.+]] = memref.view %[[ldsB]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E5M2>, #gpu.address_space<workgroup>>
  // CHECK: %[[viewBStoreMB:.+]] = rock.extract_multibuffer(%[[viewBStore2]])
  // CHECK: %[[viewBStoreTr0:.+]] = rock.transform %[[viewBStoreMB]]
  // CHECK: %[[viewBStoreTr1:.+]] = rock.transform %[[viewBStoreTr0]]
  // CHECK: %[[viewBStoreTr2:.+]] = rock.transform %[[viewBStoreTr1]]
  // CHECK: %[[viewBStoreTr3:.+]] = rock.transform %[[viewBStoreTr2]]


  // CHECK: rock.threadwise_write_all {{.*}} -> [](%[[viewBStoreTr3]])
  // CHECK: %[[viewAGemmMB:.+]] = rock.extract_multibuffer(%[[viewAGemm]])
  // CHECK: %[[viewBGemmMB:.+]] = rock.extract_multibuffer(%[[viewBGemm]])

  // CHECK: rock.blockwise_gemm_accel
  // CHECK-SAME: from %[[viewAGemmMB]]
  // CHECK-SAME: from %[[viewBGemmMB]]
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) {blockSize = 256 : i32, gridSize = 900 : i32, params = #xdlops_gemm_params1a} : memref<1x128x128xf8E4M3FN>, memref<1x128x115200xf8E5M2>, memref<1x128x115200xf32>
  return
}

// -----

#xdlops_gemm_params2 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 64, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>
// CHECK: @chiplet_grid
func.func @chiplet_grid(%arg0: memref<1x32x128xf32>, %arg1: memref<1x32x256xf32>, %arg2: memref<1x128x256xf32>) attributes {block_size = 256 : i32, grid_size = 8 : i32, arch = "amdgcn-amd-amdhsa:gfx942", numCU = 228 : i32} {
  // CHECK: %[[BID:.+]] = rock.workgroup_id
  // CHECK-DAG: %[[CHIPLET_GRP_ID:.+]] = arith.remui %[[BID]], %c4 : index
  // CHECK-DAG: %[[CHIPLET_BID:.+]] = arith.divui %[[BID]], %c4 : index
  // CHECK-DAG: %[[CHIPLET_GRP_ID_LSHIFT:.+]] = arith.muli %[[CHIPLET_GRP_ID]], %c2 : index
  // CHECK-DAG: %[[MAYBE_NEW_BID:.+]] = arith.addi %[[CHIPLET_BID]], %[[CHIPLET_GRP_ID_LSHIFT]] : index
  // CHECK-DAG: %[[IS_TAIL_BID:.+]] = arith.cmpi sgt, %[[BID]], %c7 : index
  // CHECK-DAG: %[[NEW_BID:.+]] = arith.select %[[IS_TAIL_BID]], %[[BID]], %[[MAYBE_NEW_BID]] : index
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) {blockSize = 256 : i32, gridSize = 900 : i32, params = #xdlops_gemm_params2} : memref<1x32x128xf32>, memref<1x32x256xf32>, memref<1x128x256xf32>
  return
}

// -----

#xdlops_gemm_params_double_buffer = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kpack = 8, mPerWave = 64, nPerWave = 64, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, forceUnroll = true>
// CHECK-LABEL: @fp8_bf8_xdlops_ocp_double_buffer
func.func @fp8_bf8_xdlops_ocp_double_buffer(%arg0: memref<1x128x128xf8E4M3FN>, %arg1: memref<1x128x115200xf8E5M2>, %arg2: memref<1x128x115200xf32>) attributes {block_size = 256 : i32, grid_size = 900 : i32, arch = "amdgcn-amd-amdhsa:gfx950", numCU = 256 : i32} {
  // The tuning testcase leads to padded buffers, we simplify here.
  // CHECK: %[[ldsA:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ldsB:.+]] = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>

  // CHECK-DAG: %[[viewAStore2:.+]] = memref.view %[[ldsA]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E4M3FN>, #gpu.address_space<workgroup>>
  // CHECK: %[[viewAStoreMB:.+]] = rock.extract_multibuffer(%[[viewAStore2]])
  // CHECK: %[[viewAStoreTr0:.+]] = rock.transform %[[viewAStoreMB]]
  // CHECK: %[[viewAStoreTr1:.+]] = rock.transform %[[viewAStoreTr0]]
  // CHECK: %[[viewAStoreTr2:.+]] = rock.transform %[[viewAStoreTr1]]
  // CHECK: %[[viewAStoreTr3:.+]] = rock.transform %[[viewAStoreTr2]]
  // CHECK: rock.threadwise_write_all {{.*}} -> [](%[[viewAStoreTr3]])

  // CHECK-DAG: %[[viewBStore2:.+]] = memref.view %[[ldsB]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E5M2>, #gpu.address_space<workgroup>>
  // CHECK: %[[viewBStoreMB:.+]] = rock.extract_multibuffer(%[[viewBStore2]])
  // CHECK: %[[viewBStoreTr0:.+]] = rock.transform %[[viewBStoreMB]]
  // CHECK: %[[viewBStoreTr1:.+]] = rock.transform %[[viewBStoreTr0]]
  // CHECK: %[[viewBStoreTr2:.+]] = rock.transform %[[viewBStoreTr1]]
  // CHECK: %[[viewBStoreTr3:.+]] = rock.transform %[[viewBStoreTr2]]

  // CHECK: rock.threadwise_write_all {{.*}} -> [](%[[viewBStoreTr3]])
  // CHECK-DAG: %[[viewAGemm2:.+]] = memref.view %[[ldsA]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E4M3FN>, #gpu.address_space<workgroup>>
  // CHECK: %[[viewAGemmMB:.+]] = rock.extract_multibuffer(%[[viewAGemm2]])
  // CHECK: %[[viewAGemmMBView0:.+]] = rock.transform %[[viewAGemmMB]]
  // CHECK: %[[viewAGemmMBView1:.+]] = rock.transform %[[viewAGemmMBView0]]
  // CHECK: %[[viewAGemmMBView2:.+]] = rock.transform %[[viewAGemmMBView1]]
  // CHECK: %[[viewAGemmMBView3:.+]] = rock.transform %[[viewAGemmMBView2]]
  // CHECK: %[[viewAGemmMBView4:.+]] = rock.transform %[[viewAGemmMBView3]]
  // CHECK: rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%[[viewAGemmMBView4]])

  // CHECK-DAG: %[[viewBGemm2:.+]] = memref.view %[[ldsB]][{{.*}}][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<1024xvector<8xf8E5M2>, #gpu.address_space<workgroup>>
  // CHECK: %[[viewBGemmMB:.+]] = rock.extract_multibuffer(%[[viewBGemm2]])
  // CHECK: %[[viewBGemmMBView0:.+]] = rock.transform %[[viewBGemmMB]]
  // CHECK: %[[viewBGemmMBView1:.+]] = rock.transform %[[viewBGemmMBView0]]
  // CHECK: %[[viewBGemmMBView2:.+]] = rock.transform %[[viewBGemmMBView1]]
  // CHECK: %[[viewBGemmMBView3:.+]] = rock.transform %[[viewBGemmMBView2]]
  // CHECK: %[[viewBGemmMBView4:.+]] = rock.transform %[[viewBGemmMBView3]]
  // CHECK: rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%[[viewBGemmMBView4]])
  
  // CHECK: rock.blockwise_gemm_accel 
  // CHECK-NOT: from
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) {blockSize = 256 : i32, gridSize = 900 : i32, params = #xdlops_gemm_params_double_buffer} : memref<1x128x128xf8E4M3FN>, memref<1x128x115200xf8E5M2>, memref<1x128x115200xf32>
  return
}

// -----

// Tests for scaled GEMM (FP4 with scales)
#xdlops_gemm_params_scaled = #rock.xdlops_gemm_derived_params<kpackPerBlock = 16, mPerBlock = 16, nPerBlock = 16, kpack = 32, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>

// CHECK-LABEL: @scaled_gemm_fp4_basic
func.func @scaled_gemm_fp4_basic(%arg0: memref<1x512x16xf4E2M1FN>, %arg1: memref<1x512x16xf4E2M1FN>, %arg2: memref<1x16x16xf32>, %scaleA: memref<1x512x16xf8E8M0FNU>, %scaleB: memref<1x512x16xf8E8M0FNU>) attributes {block_size = 64 : i32, grid_size = 1 : i32, kernel, arch = "amdgcn-amd-amdhsa:gfx950", num_cu = 256 : i64} {
  // Comprehensive test for scaled GEMM lowering to blockwise operations
  
  // 1. Verify LDS allocations for matrices (2x 4096 bytes for f4E2M1FN)
  // CHECK-DAG: rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  
  // 2. Verify LDS allocations for scales (2x 8192 bytes for f8E8M0FNU) 
  // CHECK-DAG: rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  
  // 3. Verify views for f4E2M1FN matrices
  // CHECK-DAG: memref.view{{.*}}: memref<4096xi8, #gpu.address_space<workgroup>> to memref<256xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
  // CHECK-DAG: memref.view{{.*}}: memref<4096xi8, #gpu.address_space<workgroup>> to memref<256xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
  
  // 4. Verify views for f8E8M0FNU scales
  // CHECK-DAG: memref.view{{.*}}: memref<8192xi8, #gpu.address_space<workgroup>> to memref<256xvector<32xf8E8M0FNU>, #gpu.address_space<workgroup>>
  // CHECK-DAG: memref.view{{.*}}: memref<8192xi8, #gpu.address_space<workgroup>> to memref<256xvector<32xf8E8M0FNU>, #gpu.address_space<workgroup>>
  
  // 5. Verify register allocations for matrix tiles
  // CHECK-DAG: rock.alloc() : memref<4xvector<32xf4E2M1FN>, #gpu.address_space<private>>
  // CHECK-DAG: rock.alloc() : memref<4xvector<32xf4E2M1FN>, #gpu.address_space<private>>
  
  // 6. Verify accumulator register
  // CHECK-DAG: rock.alloc() : memref<1xvector<4xf32>, #gpu.address_space<private>>
  
  // 7. Verify register allocations for scale tiles
  // CHECK-DAG: rock.alloc() : memref<4xvector<32xf8E8M0FNU>, #gpu.address_space<private>>
  // CHECK-DAG: rock.alloc() : memref<4xvector<32xf8E8M0FNU>, #gpu.address_space<private>>
  
  // 8. Verify threadwise_write_all for matrices
  // CHECK-DAG: rock.threadwise_write_all {{.*}} : memref<{{.*}}xf4E2M1FN, #gpu.address_space<private>> -> memref<{{.*}}xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.threadwise_write_all {{.*}} : memref<{{.*}}xf4E2M1FN, #gpu.address_space<private>> -> memref<{{.*}}xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
  
  // 9. Verify threadwise_write_all for scales
  // CHECK-DAG: rock.threadwise_write_all {{.*}} : memref<{{.*}}xf8E8M0FNU, #gpu.address_space<private>> -> memref<{{.*}}xvector<32xf8E8M0FNU>, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.threadwise_write_all {{.*}} : memref<{{.*}}xf8E8M0FNU, #gpu.address_space<private>> -> memref<{{.*}}xvector<32xf8E8M0FNU>, #gpu.address_space<workgroup>>
  
  // 10. Verify extract_multibuffer for matrices
  // CHECK-DAG: rock.extract_multibuffer({{.*}}) {{.*}}: memref<256xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.extract_multibuffer({{.*}}) {{.*}}: memref<256xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
  
  // 11. Verify extract_multibuffer for scales
  // CHECK-DAG: rock.extract_multibuffer({{.*}}) {{.*}}: memref<256xvector<32xf8E8M0FNU>, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.extract_multibuffer({{.*}}) {{.*}}: memref<256xvector<32xf8E8M0FNU>, #gpu.address_space<workgroup>>
  
  // 12. Verify blockwise_gemm_accel is called with scaled operands
  // CHECK: rock.blockwise_gemm_accel {{.*}} scaled by {{.*}} from {{.*}} * {{.*}} scaled by {{.*}} from {{.*}} features = mfma
  
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2, %scaleA, %scaleB) storeMethod( set) features =  mfma {blockSize = 64 : i32, gridSize = 1 : i32, params = #xdlops_gemm_params_scaled} : memref<1x512x16xf4E2M1FN>, memref<1x512x16xf4E2M1FN>, memref<1x16x16xf32>, memref<1x512x16xf8E8M0FNU>, memref<1x512x16xf8E8M0FNU>
  return
}

// -----

// Test scaled GEMM with different dimensions
#xdlops_gemm_params_scaled2 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 16, mPerBlock = 32, nPerBlock = 32, kpack = 32, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>

// CHECK-LABEL: @scaled_gemm_fp4_larger
func.func @scaled_gemm_fp4_larger(%arg0: memref<1x512x32xf4E2M1FN>, %arg1: memref<1x512x32xf4E2M1FN>, %arg2: memref<1x32x32xf32>, %scaleA: memref<1x512x32xf8E8M0FNU>, %scaleB: memref<1x512x32xf8E8M0FNU>) attributes {block_size = 256 : i32, grid_size = 1 : i32, kernel, arch = "amdgcn-amd-amdhsa:gfx950", num_cu = 256 : i64} {
  // Test with larger block size (256 vs 64) and dimensions (32x32 vs 16x16)
  
  // 1. Verify 4 LDS allocations (2 for matrices,  2 for scales)
  // CHECK-DAG: rock.alloc() : memref<{{[0-9]+}}xi8, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.alloc() : memref<{{[0-9]+}}xi8, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.alloc() : memref<{{[0-9]+}}xi8, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.alloc() : memref<{{[0-9]+}}xi8, #gpu.address_space<workgroup>>
  
  // 2. Verify views for matrices
  // CHECK-DAG: memref.view{{.*}}: memref<{{[0-9]+}}xi8, #gpu.address_space<workgroup>> to memref<{{[0-9]+}}xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
  // CHECK-DAG: memref.view{{.*}}: memref<{{[0-9]+}}xi8, #gpu.address_space<workgroup>> to memref<{{[0-9]+}}xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
  
  // 3. Verify views for scales
  // CHECK-DAG: memref.view{{.*}}: memref<{{[0-9]+}}xi8, #gpu.address_space<workgroup>> to memref<{{[0-9]+}}xvector<32xf8E8M0FNU>, #gpu.address_space<workgroup>>
  // CHECK-DAG: memref.view{{.*}}: memref<{{[0-9]+}}xi8, #gpu.address_space<workgroup>> to memref<{{[0-9]+}}xvector<32xf8E8M0FNU>, #gpu.address_space<workgroup>>
  
  // 4. Verify register allocations for larger tiles (8 elements vs 4)
  // CHECK-DAG: rock.alloc() : memref<8xvector<32xf4E2M1FN>, #gpu.address_space<private>>
  // CHECK-DAG: rock.alloc() : memref<8xvector<32xf4E2M1FN>, #gpu.address_space<private>>
  // CHECK-DAG: rock.alloc() : memref<1xvector<16xf32>, #gpu.address_space<private>>
  
  // 5. Verify scale register allocations (larger for 32x32)
  // CHECK-DAG: rock.alloc() : memref<8xvector<32xf8E8M0FNU>, #gpu.address_space<private>>
  // CHECK-DAG: rock.alloc() : memref<8xvector<32xf8E8M0FNU>, #gpu.address_space<private>>
  
  // 6. Verify threadwise_write_all for matrices
  // CHECK-DAG: rock.threadwise_write_all {{.*}} : memref<{{.*}}xf4E2M1FN, #gpu.address_space<private>> -> memref<{{.*}}xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.threadwise_write_all {{.*}} : memref<{{.*}}xf4E2M1FN, #gpu.address_space<private>> -> memref<{{.*}}xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
  
  // 7. Verify threadwise_write_all for scales
  // CHECK-DAG: rock.threadwise_write_all {{.*}} : memref<{{.*}}xf8E8M0FNU, #gpu.address_space<private>> -> memref<{{.*}}xvector<32xf8E8M0FNU>, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.threadwise_write_all {{.*}} : memref<{{.*}}xf8E8M0FNU, #gpu.address_space<private>> -> memref<{{.*}}xvector<32xf8E8M0FNU>, #gpu.address_space<workgroup>>
  
  // 8. Verify extract_multibuffer for matrices
  // CHECK-DAG: rock.extract_multibuffer({{.*}}) {{.*}}: memref<{{[0-9]+}}xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.extract_multibuffer({{.*}}) {{.*}}: memref<{{[0-9]+}}xvector<32xf4E2M1FN>, #gpu.address_space<workgroup>>
  
  // 9. Verify extract_multibuffer for scales
  // CHECK-DAG: rock.extract_multibuffer({{.*}}) {{.*}}: memref<{{[0-9]+}}xvector<32xf8E8M0FNU>, #gpu.address_space<workgroup>>
  // CHECK-DAG: rock.extract_multibuffer({{.*}}) {{.*}}: memref<{{[0-9]+}}xvector<32xf8E8M0FNU>, #gpu.address_space<workgroup>>
  
  // 10. Verify blockwise_gemm_accel with scaled operands
  // CHECK: rock.blockwise_gemm_accel {{.*}} scaled by {{.*}} from {{.*}} * {{.*}} scaled by {{.*}} from {{.*}} features = mfma
  
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2, %scaleA, %scaleB) storeMethod( set) features =  mfma {blockSize = 256 : i32, gridSize = 1 : i32, params = #xdlops_gemm_params_scaled2} : memref<1x512x32xf4E2M1FN>, memref<1x512x32xf4E2M1FN>, memref<1x32x32xf32>, memref<1x512x32xf8E8M0FNU>, memref<1x512x32xf8E8M0FNU>
  return
}
