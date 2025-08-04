// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @gridwise_gemm_i32_wants_i8(%a: memref<1x16x16xf32>,
                        %b: memref<1x16x16xf32>,
                        %c: memref<1x16x16xi32>) {
  // expected-error@+1 {{'rock.gridwise_gemm_accel' op floating-point input type 'f32' requires a floating-point output type, but the output type is 'i32'}}
  rock.gridwise_gemm_accel(%a, %b, %c) storeMethod(set) features = mfma|dot|atomic_add|atomic_add_f16 {
    arch = "amdgcn-amd-amdhsa:gfx908",
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #rock.xdlops_gemm_derived_params<
      kpackPerBlock = 4,
      kpack = 4,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<1x16x16xf32>, memref<1x16x16xf32>, memref<1x16x16xi32>
  func.return
}

// -----

func.func @gridwise_gemm_i8_wants_i32(%a: memref<1x16x16xi8>,
                        %b: memref<1x16x16xi8>,
                        %c: memref<1x16x16xf32>) {
  // expected-error@+1 {{'rock.gridwise_gemm_accel' op integer input type 'i8' requires an integer output type, but the output type is 'f32'}}
  rock.gridwise_gemm_accel(%a, %b, %c) storeMethod(set) features = mfma|dot|atomic_add|atomic_add_f16 {
    arch = "amdgcn-amd-amdhsa:gfx908",
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #rock.xdlops_gemm_derived_params<
      kpackPerBlock = 4,
      kpack = 4,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<1x16x16xi8>, memref<1x16x16xi8>, memref<1x16x16xf32>
  func.return
}

// -----

func.func @gridwise_gemm_m_too_big(%a: memref<1x1x2147483648xf32>,
                        %b: memref<1x1x1xf32>,
                        %c: memref<1x2147483648x1xf32>) {
  // expected-error@+1 {{'rock.gridwise_gemm_accel' op M dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  rock.gridwise_gemm_accel(%a, %b, %c) storeMethod(set) features = mfma|dot|atomic_add|atomic_add_f16 {
    arch = "amdgcn-amd-amdhsa:gfx908",
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #rock.xdlops_gemm_derived_params<
      kpackPerBlock = 4,
      kpack = 4,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<1x1x2147483648xf32>, memref<1x1x1xf32>, memref<1x2147483648x1xf32>
  func.return
}

// -----

func.func @gridwise_gemm_k_too_big(%a: memref<1x2147483648x1xf32>,
                        %b: memref<1x2147483648x1xf32>,
                        %c: memref<1x1x1xf32>) {
  // expected-error@+1 {{'rock.gridwise_gemm_accel' op K dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  rock.gridwise_gemm_accel(%a, %b, %c) storeMethod(set) features = mfma|dot|atomic_add|atomic_add_f16 {
    arch = "amdgcn-amd-amdhsa:gfx908",
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #rock.xdlops_gemm_derived_params<
      kpackPerBlock = 4,
      kpack = 4,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<1x2147483648x1xf32>, memref<1x2147483648x1xf32>, memref<1x1x1xf32>
  func.return
}
// -----

func.func @gridwise_gemm_m_too_big(%a: memref<1x1x1xf32>,
                        %b: memref<1x1x2147483648xf32>,
                        %c: memref<1x1x2147483648xf32>) {
  // expected-error@+1 {{'rock.gridwise_gemm_accel' op N dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  rock.gridwise_gemm_accel(%a, %b, %c) storeMethod(set) features = mfma|dot|atomic_add|atomic_add_f16 {
    arch = "amdgcn-amd-amdhsa:gfx908",
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #rock.xdlops_gemm_derived_params<
      kpackPerBlock = 4,
      kpack = 4,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<1x1x1xf32>, memref<1x1x2147483648xf32>, memref<1x1x2147483648xf32>
  func.return
}
