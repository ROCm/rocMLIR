// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// Valid: f32, matching shapes, group/M/K/N in int32 range
//===----------------------------------------------------------------------===//
func.func @valid_f32() {
  // CHECK: rock.gridwise_gemm
  %a = "test.tensor"() : () -> memref<2x4x8xf32>
  %b = "test.tensor"() : () -> memref<2x8x16xf32>
  %c = "test.tensor"() : () -> memref<2x4x16xf32>
  %op = "rock.gridwise_gemm"(%a, %b, %c) : (memref<2x4x8xf32>, memref<2x8x16xf32>, memref<2x4x16xf32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Valid: i8 input, i32 output
//===----------------------------------------------------------------------===//
func.func @valid_i8_i32() {
  // CHECK: rock.gridwise_gemm
  %a = "test.tensor"() : () -> memref<2x4x8xi8>
  %b = "test.tensor"() : () -> memref<2x8x16xi8>
  %c = "test.tensor"() : () -> memref<2x4x16xi32>
  %op = "rock.gridwise_gemm"(%a, %b, %c) : (memref<2x4x8xi8>, memref<2x8x16xi8>, memref<2x4x16xi32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: i8 input, f32 output
//===----------------------------------------------------------------------===//
func.func @invalid_i8_f32() {
  %a = "test.tensor"() : () -> memref<2x4x8xi8>
  %b = "test.tensor"() : () -> memref<2x8x16xi8>
  %c = "test.tensor"() : () -> memref<2x4x16xf32>
  // expected-error @+1 {{integer input type i8 requires an integer output type, but the output type is f32}}
  %op = "rock.gridwise_gemm"(%a, %b, %c) : (memref<2x4x8xi8>, memref<2x8x16xi8>, memref<2x4x16xf32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: f16 input, i32 output
//===----------------------------------------------------------------------===//
func.func @invalid_f16_i32() {
  %a = "test.tensor"() : () -> memref<2x4x8xf16>
  %b = "test.tensor"() : () -> memref<2x8x16xf16>
  %c = "test.tensor"() : () -> memref<2x4x16xi32>
  // expected-error @+1 {{floating-point input type f16 requires a floating-point output type, but the output type is i32}}
  %op = "rock.gridwise_gemm"(%a, %b, %c) : (memref<2x4x8xf16>, memref<2x8x16xf16>, memref<2x4x16xi32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: group dimension mismatch
//===----------------------------------------------------------------------===//
func.func @group_dim_mismatch() {
  %a = "test.tensor"() : () -> memref<2x4x8xf32>
  %b = "test.tensor"() : () -> memref<3x8x16xf32>
  %c = "test.tensor"() : () -> memref<2x4x16xf32>
  // expected-error @+1 {{Mismatched G dimensions in matrix multiply; A[0] = 2 b[0] = 3 C[0] = 2}}
  %op = "rock.gridwise_gemm"(%a, %b, %c) : (memref<2x4x8xf32>, memref<3x8x16xf32>, memref<2x4x16xf32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: M dimension mismatch
//===----------------------------------------------------------------------===//
func.func @m_dim_mismatch() {
  %a = "test.tensor"() : () -> memref<2x4x8xf32>
  %b = "test.tensor"() : () -> memref<2x8x16xf32>
  %c = "test.tensor"() : () -> memref<2x5x16xf32>
  // expected-error @+1 {{Mismatched M dimensions in matrix multiply: A[2] = 4 C[1] = 5}}
  %op = "rock.gridwise_gemm"(%a, %b, %c) : (memref<2x4x8xf32>, memref<2x8x16xf32>, memref<2x5x16xf32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: K dimension mismatch
//===----------------------------------------------------------------------===//
func.func @k_dim_mismatch() {
  %a = "test.tensor"() : () -> memref<2x4x7xf32>
  %b = "test.tensor"() : () -> memref<2x8x16xf32>
  %c = "test.tensor"() : () -> memref<2x4x16xf32>
  // expected-error @+1 {{Mismatched K dimensions in matrix multiply: A[1] = 7 B[1] = 8}}
  %op = "rock.gridwise_gemm"(%a, %b, %c) : (memref<2x4x7xf32>, memref<2x8x16xf32>, memref<2x4x16xf32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: N dimension mismatch
//===----------------------------------------------------------------------===//
func.func @n_dim_mismatch() {
  %a = "test.tensor"() : () -> memref<2x4x8xf32>
  %b = "test.tensor"() : () -> memref<2x8x15xf32>
  %c = "test.tensor"() : () -> memref<2x4x16xf32>
  // expected-error @+1 {{Mismatched N dimensions in matrix multiply: B[2] = 15 C[2] = 16}}
  %op = "rock.gridwise_gemm"(%a, %b, %c) : (memref<2x4x8xf32>, memref<2x8x15xf32>, memref<2x4x16xf32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: G dimension > int32_max
//===----------------------------------------------------------------------===//
func.func @g_too_large() {
  %a = "test.tensor"() : () -> memref<4294967296x4x8xf32>
  %b = "test.tensor"() : () -> memref<4294967296x8x16xf32>
  %c = "test.tensor"() : () -> memref<4294967296x4x16xf32>
  // expected-error @+1 {{G dimmension 4294967296 cannot be greater than int32_max 2147483647}}
  %op = "rock.gridwise_gemm"(%a, %b, %c) : (memref<4294967296x4x8xf32>, memref<4294967296x8x16xf32>, memref<4294967296x4x16xf32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: M dimension > int32_max
//===----------------------------------------------------------------------===//
func.func @m_too_large() {
  %a = "test.tensor"() : () -> memref<2x4294967296x8xf32>
  %b = "test.tensor"() : () -> memref<2x8x16xf32>
  %c = "test.tensor"() : () -> memref<2x4294967296x16xf32>
  // expected-error @+1 {{M dimmension 4294967296 cannot be greater than int32_max 2147483647}}
  %op = "rock.gridwise_gemm"(%a, %b, %c) : (memref<2x4294967296x8xf32>, memref<2x8x16xf32>, memref<2x4294967296x16xf32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: K dimension > int32_max
//===----------------------------------------------------------------------===//
func.func @k_too_large() {
  %a = "test.tensor"() : () -> memref<2x4x4294967296xf32>
  %b = "test.tensor"() : () -> memref<2x4294967296x16xf32>
  %c = "test.tensor"() : () -> memref<2x4x16xf32>
  // expected-error @+1 {{K dimmension 4294967296 cannot be greater than int32_max 2147483647}}
  %op = "rock.gridwise_gemm"(%a, %b, %c) : (memref<2x4x4294967296xf32>, memref<2x4294967296x16xf32>, memref<2x4x16xf32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: N dimension > int32_max
//===----------------------------------------------------------------------===//
func.func @n_too_large() {
  %a = "test.tensor"() : () -> memref<2x4x8xf32>
  %b = "test.tensor"() : () -> memref<2x8x4294967296xf32>
  %c = "test.tensor"() : () -> memref<2x4x4294967296xf32>
  // expected-error @+1 {{N dimmension 4294967296 cannot be greater than int32_max 2147483647}}
  %op = "rock.gridwise_gemm"(%a, %b, %c) : (memref<2x4x8xf32>, memref<2x8x4294967296xf32>, memref<2x4x4294967296xf32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: i8 input, output not i8 or i32
//===----------------------------------------------------------------------===//
func.func @i8_output_not_i8_or_i32() {
  %a = "test.tensor"() : () -> memref<2x4x8xi8>
  %b = "test.tensor"() : () -> memref<2x8x16xi8>
  %c = "test.tensor"() : () -> memref<2x4x16xi16>
  // expected-error @+1 {{i8 input requires i32 or i8 output}}
  %op = "rock.gridwise_gemm"(%a, %b, %c) : (memref<2x4x8xi8>, memref<2x8x16xi8>, memref<2x4x16xi16>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: f8 input, output not f32
//===----------------------------------------------------------------------===//
func.func @f8_output_not_f32() {
  %a = "test.tensor"() : () -> memref<2x4x8!rock.float8_e4m3>
  %b = "test.tensor"() : () -> memref<2x8x16!rock.float8_e4m3>
  %c = "test.tensor"() : () -> memref<2x4x16xf16>
  // expected-error @+1 {{8-bit float input requires f32 output}}
  %op = "rock.gridwise_gemm"(%a, %b, %c) : (memref<2x4x8!rock.float8_e4m3>, memref<2x8x16!rock.float8_e4m3>, memref<2x4x16xf16>) -> ()
  return
}