// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @gemm_too_big(%a: memref<2048x2048x2048xf32>,
                        %b: memref<2048x2048x2048xf32>,
                        %c: memref<2048x2048x2048xf32>) {
  // expected-error@+1 {{'rock.gemm' op underlying storage for matrix A cannot potentially be 4 GB or more}}
  rock.gemm %c = %a * %b features = dot storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx1030",
    numCu = 64 : i32}
  : memref<2048x2048x2048xf32> = memref<2048x2048x2048xf32> * memref<2048x2048x2048xf32>
  func.return
}

// -----

func.func @gemm_c_big(%a: memref<2048x2048x1xf32>,
                      %b: memref<2048x1x2048xf32>,
                      %c: memref<2048x2048x2048xf32>) {
  // expected-error@+1 {{'rock.gemm' op underlying storage for matrix C cannot potentially be 4 GB or more}}
  rock.gemm %c = %a * %b features = dot storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx1030",
    numCu = 64 : i32}
  : memref<2048x2048x2048xf32> = memref<2048x2048x1xf32> * memref<2048x1x2048xf32>
  func.return
}

// -----

func.func @gemm_c_too_big(%a: memref<2048x2048x2048xf32>,
                        %b: memref<2048x2048x2048xf32>,
                        %c: memref<2048x2048x2048xf32>) {
  // expected-error@+1 {{'rock.gemm' op underlying storage for matrix A cannot potentially be 4 GB or more}}
  rock.gemm %c = %a * %b features = dot storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx1030",
    numCu = 64 : i32}
  : memref<2048x2048x2048xf32> = memref<2048x2048x2048xf32> * memref<2048x2048x2048xf32>
  func.return
}

// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1>
func.func @gridwise_gemm_no_mixed_ab(%a: memref<1x16x16xf16>,
                        %b: memref<1x16x16xf32>,
                        %c: memref<1x16x16xf32>) {
  // expected-error@+1 {{'rock.gridwise_gemm' op cannot mix element types for A and B}}
  rock.gridwise_gemm %c = %a * %b features = dot {
    gridSize = 1 : i32,
    numCu = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x16x16xf32> = memref<1x16x16xf16> * memref<1x16x16xf32>
  func.return
}

// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1>
func.func @gridwise_gemm_i32_wants_i8(%a: memref<1x16x16xf32>,
                        %b: memref<1x16x16xf32>,
                        %c: memref<1x16x16xi32>) {
  // expected-error@+1 {{'rock.gridwise_gemm' op i32 output requires i8 input}}
  rock.gridwise_gemm %c = %a * %b features = dot {
    gridSize = 1 : i32,
    numCu = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x16x16xi32> = memref<1x16x16xf32> * memref<1x16x16xf32>
  func.return
}

// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1>
func.func @gridwise_gemm_i8_wants_i32(%a: memref<1x16x16xi8>,
                        %b: memref<1x16x16xi8>,
                        %c: memref<1x16x16xf32>) {
  // expected-error@+1 {{'rock.gridwise_gemm' op i8 input requires i32 output}}
  rock.gridwise_gemm %c = %a * %b features = dot {
    gridSize = 1 : i32,
    numCu = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x16x16xf32> = memref<1x16x16xi8> * memref<1x16x16xi8>
  func.return
}

// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1>
func.func @gridwise_gemm_m_too_big(%a: memref<1x1x2147483648xf32>,
                        %b: memref<1x1x1xf32>,
                        %c: memref<1x2147483648x1xf32>) {
  // expected-error@+1 {{'rock.gridwise_gemm' op M dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  rock.gridwise_gemm %c = %a * %b features = dot {
    gridSize = 1 : i32,
    numCu = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x2147483648x1xf32> = memref<1x1x2147483648xf32> * memref<1x1x1xf32>
  func.return
}

// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1>
func.func @gridwise_gemm_k_too_big(%a: memref<1x2147483648x1xf32>,
                        %b: memref<1x2147483648x1xf32>,
                        %c: memref<1x1x1xf32>) {
  // expected-error@+1 {{'rock.gridwise_gemm' op K dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  rock.gridwise_gemm %c = %a * %b features = dot {
    gridSize = 1 : i32,
    numCu = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x1x1xf32> = memref<1x2147483648x1xf32> * memref<1x2147483648x1xf32>
  func.return
}
// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1>
func.func @gridwise_gemm_m_too_big(%a: memref<1x1x1xf32>,
                        %b: memref<1x1x2147483648xf32>,
                        %c: memref<1x1x2147483648xf32>) {
  // expected-error@+1 {{'rock.gridwise_gemm' op N dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  rock.gridwise_gemm %c = %a * %b features = dot {
    gridSize = 1 : i32,
    numCu = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x1x2147483648xf32> = memref<1x1x1xf32> * memref<1x1x2147483648xf32>
  func.return
}
