// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// Valid: float input/output, matching shapes, no transpose, group=1
//===----------------------------------------------------------------------===//
func.func @valid_float() {
  // CHECK: rock.gemm
  %a = "test.tensor"() : () -> tensor<4x8xf32>
  %b = "test.tensor"() : () -> tensor<8x16xf32>
  %c = "test.tensor"() : () -> tensor<4x16xf32>
  %gemm = "rock.gemm"(%a, %b, %c) : (tensor<4x8xf32>, tensor<8x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
  return
}

//===----------------------------------------------------------------------===//
// Valid: int8 input, i32 output, matching shapes, group=2
//===----------------------------------------------------------------------===//
func.func @valid_int8_i32() {
  // CHECK: rock.gemm
  %a = "test.tensor"() : () -> tensor<2x4x8xi8>
  %b = "test.tensor"() : () -> tensor<2x8x16xi8>
  %c = "test.tensor"() : () -> tensor<2x4x16xi32>
  %gemm = "rock.gemm"(%a, %b, %c) : (tensor<2x4x8xi8>, tensor<2x8x16xi8>, tensor<2x4x16xi32>) -> tensor<2x4x16xi32>
  return
}

//===----------------------------------------------------------------------===//
// Invalid: float input, integer output
//===----------------------------------------------------------------------===//
func.func @float_input_int_output() {
  %a = "test.tensor"() : () -> tensor<4x8xf32>
  %b = "test.tensor"() : () -> tensor<8x16xf32>
  %c = "test.tensor"() : () -> tensor<4x16xi32>
  // expected-error @+1 {{float-valued inputs must have a floating-point output type}}
  %gemm = "rock.gemm"(%a, %b, %c) : (tensor<4x8xf32>, tensor<8x16xf32>, tensor<4x16xi32>) -> tensor<4x16xi32>
  return
}

//===----------------------------------------------------------------------===//
// Invalid: group dimensions mismatch
//===----------------------------------------------------------------------===//
func.func @group_dim_mismatch() {
  %a = "test.tensor"() : () -> tensor<2x4x8xf32>
  %b = "test.tensor"() : () -> tensor<3x8x16xf32>
  %c = "test.tensor"() : () -> tensor<2x4x16xf32>
  // expected-error @+1 {{group dimensions don't match}}
  %gemm = "rock.gemm"(%a, %b, %c) : (tensor<2x4x8xf32>, tensor<3x8x16xf32>, tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
  return
}

//===----------------------------------------------------------------------===//
// Invalid: M dimensions mismatch
//===----------------------------------------------------------------------===//
func.func @m_dim_mismatch() {
  %a = "test.tensor"() : () -> tensor<2x4x8xf32>
  %b = "test.tensor"() : () -> tensor<2x8x16xf32>
  %c = "test.tensor"() : () -> tensor<2x5x16xf32>
  // expected-error @+1 {{M dimensions don't match}}
  %gemm = "rock.gemm"(%a, %b, %c) : (tensor<2x4x8xf32>, tensor<2x8x16xf32>, tensor<2x5x16xf32>) -> tensor<2x5x16xf32>
  return
}

//===----------------------------------------------------------------------===//
// Invalid: N dimensions mismatch
//===----------------------------------------------------------------------===//
func.func @n_dim_mismatch() {
  %a = "test.tensor"() : () -> tensor<2x4x8xf32>
  %b = "test.tensor"() : () -> tensor<2x8x16xf32>
  %c = "test.tensor"() : () -> tensor<2x4x15xf32>
  // expected-error @+1 {{N dimensions don't match}}
  %gemm = "rock.gemm"(%a, %b, %c) : (tensor<2x4x8xf32>, tensor<2x8x16xf32>, tensor<2x4x15xf32>) -> tensor<2x4x15xf32>
  return
}

//===----------------------------------------------------------------------===//
// Invalid: K dimensions mismatch
//===----------------------------------------------------------------------===//
func.func @k_dim_mismatch() {
  %a = "test.tensor"() : () -> tensor<2x4x7xf32>
  %b = "test.tensor"() : () -> tensor<2x8x16xf32>
  %c = "test.tensor"() : () -> tensor<2x4x16xf32>
  // expected-error @+1 {{K dimensions don't match}}
  %gemm = "rock.gemm"(%a, %b, %c) : (tensor<2x4x7xf32>, tensor<2x8x16xf32>, tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
  return
}

//===----------------------------------------------------------------------===//
// Invalid: xdlops GEMM with non-xdlops tuning parameters
//===----------------------------------------------------------------------===//
func.func @xdlops_non_xdlops_params() {
  %a = "test.tensor"() : () -> tensor<2x4x8xf16>
  %b = "test.tensor"() : () -> tensor<2x8x16xf16>
  %c = "test.tensor"() : () -> tensor<2x4x16xf16>
  // expected-error @+1 {{an xdlops GEMM has non-xdlops tuning parameters}}
  %gemm = "rock.gemm"(%a, %b, %c) {features = #rock.gemm_features<mfma>, params = #rock.general_gemm_params<>} : (tensor<2x4x8xf16>, tensor<2x8x16xf16>, tensor<2x4x16xf16>) -> tensor<2x4x16xf16>
  return
}

//===----------------------------------------------------------------------===//
// Invalid: all-hardware GEMM with non-general tuning parameters
//===----------------------------------------------------------------------===//
func.func @all_hw_non_general_params() {
  %a = "test.tensor"() : () -> tensor<2x4x8xf16>
  %b = "test.tensor"() : () -> tensor<2x8x16xf16>
  %c = "test.tensor"() : () -> tensor<2x4x16xf16>
  // expected-error @+1 {{an all-hardware gemm must used the general gemm tuning parameters}}
  %gemm = "rock.gemm"(%a, %b, %c) {features = #rock.gemm_features<none>, params = #rock.xdlops_gemm_params<>} : (tensor<2x4x8xf16>, tensor<2x8x16xf16>, tensor<2x4x16xf16>) -> tensor<2x4x16xf16>
  return
}

//===----------------------------------------------------------------------===//
// Invalid: derivedBlockSize with generalGemmParams
//===----------------------------------------------------------------------===//
func.func @derived_block_size_with_general_params() {
  %a = "test.tensor"() : () -> tensor<2x4x8xf16>
  %b = "test.tensor"() : () -> tensor<2x8x16xf16>
  %c = "test.tensor"() : () -> tensor<2x4x16xf16>
  // expected-error @+1 {{cannot have derivedBlockSize when gemm has generalGemmParams}}
  %gemm = "rock.gemm"(%a, %b, %c) {params = #rock.general_gemm_params<>, derivedBlockSize = 64 : i32} : (tensor<2x4x8xf16>, tensor<2x8x16xf16>, tensor<2x4x16xf16>) -> tensor<2x4x16xf16>
  return
}

//===----------------------------------------------------------------------===//
// Invalid: derivedBlockSize with non-xdlops/wmma features
//===----------------------------------------------------------------------===//
func.func @derived_block_size_non_xdlops() {
  %a = "test.tensor"() : () -> tensor<2x4x8xf16>
  %b = "test.tensor"() : () -> tensor<2x8x16xf16>
  %c = "test.tensor"() : () -> tensor<2x4x16xf16>
  // expected-error @+1 {{general gemm kernels shouldn't have derived block size.}}
  %gemm = "rock.gemm"(%a, %b, %c) {derivedBlockSize = 64 : i32} : (tensor<2x4x8xf16>, tensor<2x8x16xf16>, tensor<2x4x16xf16>) -> tensor<2x4x16xf16>
  return
}

//===----------------------------------------------------------------------===//
// Valid: xdlops GEMM with xdlops tuning parameters
//===----------------------------------------------------------------------===//
func.func @xdlops_valid() {
  // CHECK: rock.gemm
  %a = "test.tensor"() : () -> tensor<2x4x8xf16>
  %b = "test.tensor"() : () -> tensor<2x8x16xf16>
  %c = "test.tensor"() : () -> tensor<2x4x16xf16>
  %gemm = "rock.gemm"(%a, %b, %c) {features = #rock.gemm_features<mfma>, params = #rock.xdlops_gemm_params<>} : (tensor<2x4x8xf16>, tensor<2x8x16xf16>, tensor<2x4x16xf16>) -> tensor<2x4x16xf16>
  return
}

//===----------------------------------------------------------------------===//
// Valid: all-hardware GEMM with general tuning parameters
//===----------------------------------------------------------------------===//
func.func @all_hw_valid() {
  // CHECK: rock.gemm
  %a = "test.tensor"() : () -> tensor<2x4x8xf16>
  %b = "test.tensor"() : () -> tensor<2x8x16xf16>
  %c = "test.tensor"() : () -> tensor<2x4x16xf16>
  %gemm = "rock.gemm"(%a, %b, %c) {features = #rock.gemm_features<none>, params = #rock.general_gemm_params<>} : (tensor<2x4x8xf16>, tensor<2x8x16xf16>, tensor<2x4x16xf16>) -> tensor<2x4x16xf16>
  return
}