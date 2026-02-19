// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
func.func @gridwise_gemm_i32_wants_i8(%a: memref<1x16x16xf32>,
                        %b: memref<1x16x16xf32>,
                        %c: memref<1x16x16xi32>) attributes {arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op floating-point input type 'f32' requires a floating-point output type, but the output type is 'i32'}}
  rock.gridwise_gemm %c = %a * %b storeMethod(set) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x16x16xi32> = memref<1x16x16xf32> * memref<1x16x16xf32>
  func.return
}

// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
func.func @gridwise_gemm_i8_wants_i32(%a: memref<1x16x16xi8>,
                        %b: memref<1x16x16xi8>,
                        %c: memref<1x16x16xf32>) attributes {arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op integer input type 'i8' requires an integer output type, but the output type is 'f32'}}
  rock.gridwise_gemm %c = %a * %b storeMethod(set) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x16x16xf32> = memref<1x16x16xi8> * memref<1x16x16xi8>
  func.return
}

// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
func.func @gridwise_gemm_m_too_big(%a: memref<1x1x2147483648xf32>,
                        %b: memref<1x1x1xf32>,
                        %c: memref<1x2147483648x1xf32>) attributes {arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op M dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  rock.gridwise_gemm %c = %a * %b storeMethod(set) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x2147483648x1xf32> = memref<1x1x2147483648xf32> * memref<1x1x1xf32>
  func.return
}

// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
func.func @gridwise_gemm_k_too_big(%a: memref<1x2147483648x1xf32>,
                        %b: memref<1x2147483648x1xf32>,
                        %c: memref<1x1x1xf32>) attributes {arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op K dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  rock.gridwise_gemm %c = %a * %b storeMethod(set) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x1x1xf32> = memref<1x2147483648x1xf32> * memref<1x2147483648x1xf32>
  func.return
}
// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
func.func @gridwise_gemm_m_too_big(%a: memref<1x1x1xf32>,
                        %b: memref<1x1x2147483648xf32>,
                        %c: memref<1x1x2147483648xf32>) attributes {arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op N dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  rock.gridwise_gemm %c = %a * %b storeMethod(set) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x1x2147483648xf32> = memref<1x1x1xf32> * memref<1x1x2147483648xf32>
  func.return
}

// -----

func.func @expand_strides_rank_mismatch(%input: tensor<4x24xf16>, %output: tensor<4x48x24xf16>) -> tensor<4x48x24xf16> {
  // expected-error@+1 {{'rock.expand_strides' op input and output must have the same rank}}
  %result = rock.expand_strides %input into %output : tensor<4x24xf16> into tensor<4x48x24xf16> -> tensor<4x48x24xf16>
  return %result : tensor<4x48x24xf16>
}

// -----

func.func @expand_strides_output_too_small(%input: tensor<4x24x24xf16>, %output: tensor<4x20x24xf16>) -> tensor<4x20x24xf16> {
  // expected-error@+1 {{'rock.expand_strides' op output dimension 20 is smaller than input dimension 24}}
  %result = rock.expand_strides %input into %output : tensor<4x24x24xf16> into tensor<4x20x24xf16> -> tensor<4x20x24xf16>
  return %result : tensor<4x20x24xf16>
}

// -----

func.func @expand_strides_element_type_mismatch(%input: tensor<4x24x24xf16>, %output: tensor<4x48x24xf32>) -> tensor<4x48x24xf32> {
  // expected-error@+1 {{'rock.expand_strides' op input and output must have the same element type}}
  %result = rock.expand_strides %input into %output : tensor<4x24x24xf16> into tensor<4x48x24xf32> -> tensor<4x48x24xf32>
  return %result : tensor<4x48x24xf32>
}

// -----

func.func @cond_barrier_invalid_type(%pred: i32) {
  // expected-error@+1 {{'rock.cond_barrier' op operand #0 must be 1-bit signless integer, but got 'i32'}}
  rock.cond_barrier %pred : i32
  return
}

