// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// COM: Negative coverage for migraphx::LiteralOp::verify and
// COM: migraphx::UnpackOp::verify in mlir/lib/Dialect/MIGraphX/IR/MIGraphX.cpp.

// COM: non-splat literal whose value shape does not match the logical shape
func.func @literal_shape_mismatch() {
  // expected-error @+1 {{non-splat literals must have a value that matches the literal's logical shape}}
  %0 = migraphx.literal (dense<[1, 2, 3, 4]> : tensor<4xi32>) : <2x2xi32, 2x1>
  return
}

// -----

// COM: non-splat literal whose strides are not in standard (row-major) form
func.func @literal_non_standard_strides() {
  // expected-error @+1 {{strides of non-splat literal are not in standard shape}}
  %0 = migraphx.literal (dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>) : <2x2xi32, 1x2>
  return
}

// -----

// COM: unpack axis must be within the input rank
func.func @unpack_axis_out_of_range(%x: !migraphx.shaped<8x2xi8, 2x1>) {
  // expected-error @+1 {{axis out of range of shape}}
  %y = migraphx.unpack %x {axis = 5 : i64} : <8x2xi8, 2x1> -> <8x4xi8, 4x1>
  return
}

// -----

// COM: for int8 unpack the output axis length must be twice the input axis length
func.func @unpack_wrong_output_length(%x: !migraphx.shaped<8x2xi8, 2x1>) {
  // expected-error @+1 {{expected length along input axis to be half the length along output axis}}
  %y = migraphx.unpack %x {axis = 1 : i64} : <8x2xi8, 2x1> -> <8x8xi8, 8x1>
  return
}
