// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// Valid: axis in range, int8 <-> int8, output dim is double input dim
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @valid_unpack
  func.func @valid_unpack() -> !migraphx.shaped<2x4xi8, 4x1> {
    %in = "migraphx.literal"() {value = dense<1> : tensor<2x2xi8>} : () -> !migraphx.shaped<2x2xi8, 2x1>
    %out = "migraphx.literal"() {value = dense<1> : tensor<2x4xi8>} : () -> !migraphx.shaped<2x4xi8, 4x1>
    %u = "migraphx.unpack"(%in, %out) {axis = 1} : (!migraphx.shaped<2x2xi8, 2x1>, !migraphx.shaped<2x4xi8, 4x1>) -> !migraphx.shaped<2x4xi8, 4x1>
    return %u : !migraphx.shaped<2x4xi8, 4x1>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: axis out of range (too high)
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{axis out of range of shape: 3}}
  func.func @axis_too_high() -> !migraphx.shaped<2x4xi8, 4x1> {
    %in = "migraphx.literal"() {value = dense<1> : tensor<2x2xi8>} : () -> !migraphx.shaped<2x2xi8, 2x1>
    %out = "migraphx.literal"() {value = dense<1> : tensor<2x4xi8>} : () -> !migraphx.shaped<2x4xi8, 4x1>
    %u = "migraphx.unpack"(%in, %out) {axis = 3} : (!migraphx.shaped<2x2xi8, 2x1>, !migraphx.shaped<2x4xi8, 4x1>) -> !migraphx.shaped<2x4xi8, 4x1>
    return %u : !migraphx.shaped<2x4xi8, 4x1>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: axis out of range (negative)
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{axis out of range of shape: -1}}
  func.func @axis_negative() -> !migraphx.shaped<2x4xi8, 4x1> {
    %in = "migraphx.literal"() {value = dense<1> : tensor<2x2xi8>} : () -> !migraphx.shaped<2x2xi8, 2x1>
    %out = "migraphx.literal"() {value = dense<1> : tensor<2x4xi8>} : () -> !migraphx.shaped<2x4xi8, 4x1>
    %u = "migraphx.unpack"(%in, %out) {axis = -1} : (!migraphx.shaped<2x2xi8, 2x1>, !migraphx.shaped<2x4xi8, 4x1>) -> !migraphx.shaped<2x4xi8, 4x1>
    return %u : !migraphx.shaped<2x4xi8, 4x1>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: int8 <-> int8, but output dim is not double input dim
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{expected length along input axis to be half the length along output axis}}
  func.func @not_double_length() -> !migraphx.shaped<2x5xi8, 5x1> {
    %in = "migraphx.literal"() {value = dense<1> : tensor<2x2xi8>} : () -> !migraphx.shaped<2x2xi8, 2x1>
    %out = "migraphx.literal"() {value = dense<1> : tensor<2x5xi8>} : () -> !migraphx.shaped<2x5xi8, 5x1>
    %u = "migraphx.unpack"(%in, %out) {axis = 1} : (!migraphx.shaped<2x2xi8, 2x1>, !migraphx.shaped<2x5xi8, 5x1>) -> !migraphx.shaped<2x5xi8, 5x1>
    return %u : !migraphx.shaped<2x5xi8, 5x1>
  }
}

//===----------------------------------------------------------------------===//
// Valid: not int8 <-> int8, skip double-length check
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @not_int8
  func.func @not_int8() -> !migraphx.shaped<2x5xf32, 5x1> {
    %in = "migraphx.literal"() {value = dense<1.0> : tensor<2x2xf32>} : () -> !migraphx.shaped<2x2xf32, 2x1>
    %out = "migraphx.literal"() {value = dense<1.0> : tensor<2x5xf32>} : () -> !migraphx.shaped<2x5xf32, 5x1>
    %u = "migraphx.unpack"(%in, %out) {axis = 1} : (!migraphx.shaped<2x2xf32, 2x1>, !migraphx.shaped<2x5xf32, 5x1>) -> !migraphx.shaped<2x5xf32, 5x1>
    return %u : !migraphx.shaped<2x5xf32, 5x1>
  }
}

//===----------------------------------------------------------------------===//
// Valid: axis = 0, int8 <-> int8, output dim is double input dim
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @axis0
  func.func @axis0() -> !migraphx.shaped<4x2xi8, 2x1> {
    %in = "migraphx.literal"() {value = dense<1> : tensor<2x2xi8>} : () -> !migraphx.shaped<2x2xi8, 2x1>
    %out = "migraphx.literal"() {value = dense<1> : tensor<4x2xi8>} : () -> !migraphx.shaped<4x2xi8, 2x1>
    %u = "migraphx.unpack"(%in, %out) {axis = 0} : (!migraphx.shaped<2x2xi8, 2x1>, !migraphx.shaped<4x2xi8, 2x1>) -> !migraphx.shaped<4x2xi8, 2x1>
    return %u : !migraphx.shaped<4x2xi8, 2x1>
  }
}