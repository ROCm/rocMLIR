// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// COM: Negative coverage for rock::ReduceOp::verify in
// COM: mlir/lib/Dialect/Rock/IR/RockDialect.cpp.

// COM: the reduction dimension must collapse to size 1
func.func @reduce_axis_not_one(%arg0: memref<2x12x12xf32>, %arg1: memref<2x12x12xf32>) {
  // expected-error @+1 {{The size of the reduction dimension should be 1.}}
  rock.reduce sum %arg0 into %arg1 {axis = 2 : index, blockSize = 64 : i32, gridSize = 2 : i32} : memref<2x12x12xf32> into memref<2x12x12xf32>
  return
}

// -----

// COM: non-reduction dimensions must match the input shape
func.func @reduce_nonaxis_mismatch(%arg0: memref<2x12x12xf32>, %arg1: memref<2x13x1xf32>) {
  // expected-error @+1 {{The size of the non-reduction dimension should match the input.}}
  rock.reduce sum %arg0 into %arg1 {axis = 2 : index, blockSize = 64 : i32, gridSize = 2 : i32} : memref<2x12x12xf32> into memref<2x13x1xf32>
  return
}

// -----

// COM: input and output element types must match (enforced by the ODS
// COM: SameElementType trait before the custom verifier runs)
func.func @reduce_elem_type_mismatch(%arg0: memref<2x12x12xf32>, %arg1: memref<2x12x1xf16>) {
  // expected-error @+1 {{failed to verify that all of {in, out} have same element type}}
  rock.reduce sum %arg0 into %arg1 {axis = 2 : index, blockSize = 64 : i32, gridSize = 2 : i32} : memref<2x12x12xf32> into memref<2x12x1xf16>
  return
}

// -----

// COM: reduce max is only supported for f32
func.func @reduce_max_non_f32(%arg0: memref<2x12x12xf16>, %arg1: memref<2x12x1xf16>) {
  // expected-error @+1 {{reduce max only supports f32}}
  rock.reduce max %arg0 into %arg1 {axis = 2 : index, blockSize = 64 : i32, gridSize = 2 : i32} : memref<2x12x12xf16> into memref<2x12x1xf16>
  return
}
