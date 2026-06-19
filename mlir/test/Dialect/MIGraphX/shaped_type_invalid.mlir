// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// COM: Negative coverage for migraphx::MIXRShapedType::parse in
// COM: mlir/lib/Dialect/MIGraphX/IR/MIGraphX.cpp.

// COM: missing dimension list / element type
func.func @shaped_missing_type(%arg: !migraphx.shaped<>) {
  // expected-error @-1 {{expected shaped dimension list with type}}
  // expected-error @-2 {{expected non-function type}}
  func.return
}

// -----

// COM: non-scalar shaped type without the stride list
func.func @shaped_missing_strides(%arg: !migraphx.shaped<4x4xf32>) {
  // expected-error @-1 {{expected `,` and a `x`-separated list in non-scalar migraphx.shaped type}}
  // expected-error @-2 {{expected ','}}
  func.return
}
