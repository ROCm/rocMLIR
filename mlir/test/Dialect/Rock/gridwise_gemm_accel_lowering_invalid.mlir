// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -verify-diagnostics %s

// -----

// Test case: LDS size exceeds architecture limit
// For gfx942, maxSharedMemPerWG = 65536 bytes
// This test allocates LDS larger than 65536 bytes
// kpackPerBlock * mPerBlock * kpack * sizeof(f32) + kpackPerBlock * nPerBlock * kpack * sizeof(f32)
// = 32 * 256 * 8 * 4 + 32 * 256 * 8 * 4 = 262144 + 262144 = 524288 bytes > 65536
// Format: A (G x K x M), B (G x K x N), C (G x M x N)
#xdlops_gemm_params_too_much_lds = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 256, nPerBlock = 256, kpack = 8, mPerWave = 64, nPerWave = 64, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>
func.func @excessive_lds_usage(%arg0: memref<1x256x256xf32>, %arg1: memref<1x256x256xf32>, %arg2: memref<1x256x256xf32>) attributes {block_size = 256 : i32, grid_size = 1 : i32, arch = "amdgcn-amd-amdhsa:gfx942", numCU = 304 : i32} {
  // expected-error @+2 {{requires too much LDS}}
  // expected-error @+1 {{failed to legalize operation 'rock.gridwise_gemm_accel'}}
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) {blockSize = 256 : i32, gridSize = 1 : i32, params = #xdlops_gemm_params_too_much_lds} : memref<1x256x256xf32>, memref<1x256x256xf32>, memref<1x256x256xf32>
  return
}

// -----

// Test case: LDS size exceeds limit for gfx950 (160KB limit)
// kpackPerBlock * mPerBlock * kpack * sizeof(f32) + kpackPerBlock * nPerBlock * kpack * sizeof(f32) > 163840
// 16 * 512 * 8 * 4 + 16 * 512 * 8 * 4 = 262144 + 262144 = 524288 bytes > 163840
// Format: A (G x K x M), B (G x K x N), C (G x M x N)
#xdlops_gemm_params_gfx950_lds = #rock.xdlops_gemm_derived_params<kpackPerBlock = 16, mPerBlock = 512, nPerBlock = 512, kpack = 8, mPerWave = 64, nPerWave = 64, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>
func.func @gfx950_lds_exceeded(%arg0: memref<1x128x512xf32>, %arg1: memref<1x128x512xf32>, %arg2: memref<1x512x512xf32>) attributes {block_size = 256 : i32, grid_size = 1 : i32, arch = "amdgcn-amd-amdhsa:gfx950", numCU = 256 : i32} {
  // expected-error @+2 {{requires too much LDS}}
  // expected-error @+1 {{failed to legalize operation 'rock.gridwise_gemm_accel'}}
  rock.gridwise_gemm_accel(%arg0, %arg1, %arg2) storeMethod( set) {blockSize = 256 : i32, gridSize = 1 : i32, params = #xdlops_gemm_params_gfx950_lds} : memref<1x128x512xf32>, memref<1x128x512xf32>, memref<1x512x512xf32>
  return
}

