// RUN: rocmlir-opt -mlir-print-local-scope -split-input-file -rock-blockwise-load-tile-to-threadwise -canonicalize -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @doublebuffer
func.func @doublebuffer(%arg0: memref<1x384x64xf32>) attributes {block_size = 256 : i32, grid_size = 900 : i32, arch = "amdgcn-amd-amdhsa:gfx942", numCU = 304 : i32} {
  %c0 = arith.constant 0 : index
  %lds = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  %reg = rock.alloc() : memref<16xf32, #gpu.address_space<private>>
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>

  // CHECK: affine.for %[[arg1:.+]] = 0 to 2
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into
    // CHECK-NEXT: rock.yield
    // CHECK: {name = "GlobalRead"}

    // CHECK: rock.stage
    // CHECK: rock.threadwise_copy
    // CHECK-NEXT: rock.threadwise_write_all
    // CHECK-NEXT: rock.yield
    // CHECK: {name = "LDSWrite"}

    // CHECK: rock.lds_barrier

    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into
    // CHECK-NEXT: rock.yield
    // CHECK: {name = "LDSRead"}
  affine.for %arg1 = 0 to 2 {
    rock.blockwise_load_tile %0[%arg1, %c0, %c0, %c0, %c0] LDS -> %lds -> %reg {elementType = f32, elementLoadType = f32, matrixParamsA = #rock.blockwise_matrix_params<elementType = f32, elementTypeLoad = f32, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 384, inDPerThread = 4>, matrixParamsB = #rock.blockwise_matrix_params<elementType = f32, elementTypeLoad = f32, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 384, inDPerThread = 4>, blockSize = 64 : i32, loadType = #rock<GemmLoadTileType DoubleBuffer>, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>} : memref<1x64x384xf32> LDS -> memref<4096xi8, #gpu.address_space<workgroup>> -> memref<16xf32, #gpu.address_space<private>>
  }
  return
}

// CHECK-LABEL: @default
func.func @default(%arg0: memref<1x384x64xf32>) attributes {block_size = 256 : i32, grid_size = 900 : i32, arch = "amdgcn-amd-amdhsa:gfx942", numCU = 304 : i32} {
  %c0 = arith.constant 0 : index
  %lds = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>

  // CHECK: affine.for %[[arg1:.+]] = 0 to 2
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into
    // CHECK-NEXT: rock.yield
    // CHECK: {name = "GlobalRead"}

    // CHECK: rock.stage
    // CHECK: rock.threadwise_copy
    // CHECK-NEXT: rock.threadwise_write_all
    // CHECK-NEXT: rock.yield
    // CHECK: {name = "LDSWrite"}
  affine.for %arg1 = 0 to 2 {
    rock.blockwise_load_tile %0[%arg1, %c0, %c0, %c0, %c0] LDS -> %lds {elementType = f32, elementLoadType = f32, matrixParamsA = #rock.blockwise_matrix_params<elementType = f32, elementTypeLoad = f32, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 384, inDPerThread = 4>, matrixParamsB = #rock.blockwise_matrix_params<elementType = f32, elementTypeLoad = f32, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 384, inDPerThread = 4>, blockSize = 64 : i32, loadType = #rock<GemmLoadTileType Default>, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>} : memref<1x64x384xf32> LDS -> memref<4096xi8, #gpu.address_space<workgroup>>
  }
  return
}

// CHECK-LABEL: @bypasslds
func.func @bypasslds(%arg0: memref<1x384x64xf32>) attributes {block_size = 256 : i32, grid_size = 900 : i32, arch = "amdgcn-amd-amdhsa:gfx942", numCU = 304 : i32} {
  %c0 = arith.constant 0 : index
  %reg = rock.alloc() : memref<16xf32, #gpu.address_space<private>>
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>

  // CHECK: affine.for %[[arg1:.+]] = 0 to 2
    // CHECK: rock.stage
    // CHECK: rock.threadwise_read_into
    // CHECK-NEXT: rock.yield
    // CHECK: {name = "GlobalRead"}

    // CHECK: rock.stage
    // CHECK: affine.for %arg2 = 0 to 1 {
      // CHECK: rock.threadwise_read_into
    // CHECK: rock.yield
    // CHECK: {name = "RegTranspose"}
  affine.for %arg1 = 0 to 2 {
    rock.blockwise_load_tile %0[%arg1, %c0, %c0, %c0, %c0] -> %reg {elementType = f32, elementLoadType = f32, matrixParamsA = #rock.blockwise_matrix_params<elementType = f32, elementTypeLoad = f32, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 384, inDPerThread = 4>, matrixParamsB = #rock.blockwise_matrix_params<elementType = f32, elementTypeLoad = f32, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 384, inDPerThread = 4>, blockSize = 64 : i32, loadType = #rock<GemmLoadTileType BypassLDS>, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>} : memref<1x64x384xf32> -> memref<16xf32, #gpu.address_space<private>>
  }
  return
}
