// RUN: rocmlir-opt %s -migraphx-tosa-simplify | FileCheck %s

// Test 1: Eliminate redundant cast (same input and output types)
// CHECK-LABEL: @eliminate_redundant_cast
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NOT: tosa.cast
// CHECK: return %[[ARG0]] : tensor<4xf32>
func.func @eliminate_redundant_cast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// ----

// Test 2: Eliminate cast chain that results in same type
// CHECK-LABEL: @eliminate_cast_chain
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NOT: tosa.cast
// CHECK: return %[[ARG0]] : tensor<4xf32>
func.func @eliminate_cast_chain(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf16>
  %1 = tosa.cast %0 : (tensor<4xf16>) -> tensor<4xi32>
  %2 = tosa.cast %1 : (tensor<4xi32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

// ----

// Test 3: Keep necessary cast (different types)
// CHECK-LABEL: @keep_necessary_cast
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf16>
// CHECK: %[[CAST:.*]] = tosa.cast %[[ARG0]] : (tensor<4xf32>) -> tensor<4xf16>
// CHECK: return %[[CAST]] : tensor<4xf16>
func.func @keep_necessary_cast(%arg0: tensor<4xf32>) -> tensor<4xf16> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// ----

// Test 4: Partial cast chain elimination
// CHECK-LABEL: @partial_cast_chain
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf16>
// CHECK: %[[CAST:.*]] = tosa.cast %[[ARG0]] : (tensor<4xf32>) -> tensor<4xf16>
// CHECK: return %[[CAST]] : tensor<4xf16>
func.func @partial_cast_chain(%arg0: tensor<4xf32>) -> tensor<4xf16> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %1 = tosa.cast %0 : (tensor<4xf32>) -> tensor<4xf16>
  return %1 : tensor<4xf16>
}

// ----

// Test 5: Multiple independent casts
// CHECK-LABEL: @multiple_independent_casts
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>, %[[ARG1:.*]]: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>)
// CHECK-NOT: tosa.cast
// CHECK: return %[[ARG0]], %[[ARG1]] : tensor<4xf32>, tensor<4xi32>
func.func @multiple_independent_casts(%arg0: tensor<4xf32>, %arg1: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %1 = tosa.cast %arg1 : (tensor<4xi32>) -> tensor<4xi32>
  return %0, %1 : tensor<4xf32>, tensor<4xi32>
}

// ----

// Test 6: Complex cast chain with intermediate operations
// CHECK-LABEL: @complex_cast_chain
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[CAST0:.*]] = tosa.cast %[[ARG0]] : (tensor<4xf32>) -> tensor<4xf16>
// CHECK: %[[ABS:.*]] = tosa.abs %[[CAST0]] : (tensor<4xf16>) -> tensor<4xf16>
// CHECK: %[[CAST1:.*]] = tosa.cast %[[ABS]] : (tensor<4xf16>) -> tensor<4xf32>
// CHECK-NOT: tosa.cast %[[CAST1]]
// CHECK: return %[[CAST1]] : tensor<4xf32>
func.func @complex_cast_chain(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf16>
  %1 = tosa.abs %0 : (tensor<4xf16>) -> tensor<4xf16>
  %2 = tosa.cast %1 : (tensor<4xf16>) -> tensor<4xf32>
  %3 = tosa.cast %2 : (tensor<4xf32>) -> tensor<4xf32>
  return %3 : tensor<4xf32>
}

// ----

// Test 7: Different tensor shapes (should not be eliminated)
// CHECK-LABEL: @different_shapes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<2x2xf32>
// CHECK: %[[CAST:.*]] = tosa.cast %[[ARG0]] : (tensor<4xf32>) -> tensor<2x2xf32>
// CHECK: return %[[CAST]] : tensor<2x2xf32>
func.func @different_shapes(%arg0: tensor<4xf32>) -> tensor<2x2xf32> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// ----

// Test 8: Long chain of same-type casts
// CHECK-LABEL: @long_redundant_chain
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NOT: tosa.cast
// CHECK: return %[[ARG0]] : tensor<4xf32>
func.func @long_redundant_chain(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %1 = tosa.cast %0 : (tensor<4xf32>) -> tensor<4xf32>
  %2 = tosa.cast %1 : (tensor<4xf32>) -> tensor<4xf32>
  %3 = tosa.cast %2 : (tensor<4xf32>) -> tensor<4xf32>
  return %3 : tensor<4xf32>
}

// ----

// Test 9: Mixed necessary and unnecessary casts
// CHECK-LABEL: @mixed_casts
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xi8>
// CHECK: %[[CAST0:.*]] = tosa.cast %[[ARG0]] : (tensor<4xf32>) -> tensor<4xi32>
// CHECK: %[[CAST1:.*]] = tosa.cast %[[CAST0]] : (tensor<4xi32>) -> tensor<4xi8>
// CHECK: return %[[CAST1]] : tensor<4xi8>
func.func @mixed_casts(%arg0: tensor<4xf32>) -> tensor<4xi8> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf32>  // redundant
  %1 = tosa.cast %0 : (tensor<4xf32>) -> tensor<4xi32>     // necessary
  %2 = tosa.cast %1 : (tensor<4xi32>) -> tensor<4xi8>      // necessary
  return %2 : tensor<4xi8>
}

// ----

// Test 10: Cast with multiple uses
// CHECK-LABEL: @cast_multiple_uses
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-NOT: tosa.cast
// CHECK: return %[[ARG0]], %[[ARG0]] : tensor<4xf32>, tensor<4xf32>
func.func @cast_multiple_uses(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0, %0 : tensor<4xf32>, tensor<4xf32>
}

// ----

// Test 11: Complex cast chain with multiple uses
// CHECK-LABEL: @complex_cast_chain_multiple_uses
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf16>)
// CHECK: %[[CAST0:.*]] = tosa.cast %[[ARG0]] : (tensor<4xf32>) -> tensor<4xf16>
// CHECK-NOT: tosa.cast %[[CAST0]]
// CHECK: %[[ABS:.*]] = tosa.abs %[[CAST0]] : (tensor<4xf16>) -> tensor<4xf16>
// CHECK: return %[[ARG0]], %[[ABS]] : tensor<4xf32>, tensor<4xf16>
func.func @complex_cast_chain_multiple_uses(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf16>) {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf16>
  %1 = tosa.cast %0 : (tensor<4xf16>) -> tensor<4xf32>
  %2 = tosa.cast %1 : (tensor<4xf32>) -> tensor<4xf32>
  %3 = tosa.abs %0 : (tensor<4xf16>) -> tensor<4xf16>
  return %2, %3: tensor<4xf32>, tensor<4xf16>
}
