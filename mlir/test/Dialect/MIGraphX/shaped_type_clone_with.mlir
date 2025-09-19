// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: cloneWith - shape only
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @clone_shape_only
// CHECK:      %[[C:.*]] = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
// CHECK:      %[[CLONE:.*]] = "test.clone_with_shape"(%[[C]]) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> !migraphx.shaped<4x5xf32, 6x2x1>
// CHECK:      return %[[CLONE]]
func.func @clone_shape_only() -> !migraphx.shaped<4x5xf32, 6x2x1> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
  %clone = "test.clone_with_shape"(%c) {shape = [4, 5]} : (!migraphx.shaped<2x3xf32, 6x2x1>) -> !migraphx.shaped<4x5xf32, 6x2x1>
  return %clone : !migraphx.shaped<4x5xf32, 6x2x1>
}

//===----------------------------------------------------------------------===//
// Test: cloneWith - strides only
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @clone_strides_only
// CHECK:      %[[C:.*]] = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
// CHECK:      %[[CLONE:.*]] = "test.clone_with_strides"(%[[C]]) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> !migraphx.shaped<2x3xf32, 12x4x1>
// CHECK:      return %[[CLONE]]
func.func @clone_strides_only() -> !migraphx.shaped<2x3xf32, 12x4x1> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
  %clone = "test.clone_with_strides"(%c) {strides = [12, 4, 1]} : (!migraphx.shaped<2x3xf32, 6x2x1>) -> !migraphx.shaped<2x3xf32, 12x4x1>
  return %clone : !migraphx.shaped<2x3xf32, 12x4x1>
}

//===----------------------------------------------------------------------===//
// Test: cloneWith - element type only
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @clone_elemtype_only
// CHECK:      %[[C:.*]] = "migraphx.literal"() {value = dense<1> : tensor<2x3xi32>} : () -> !migraphx.shaped<2x3xi32, 6x2x1>
// CHECK:      %[[CLONE:.*]] = "test.clone_with_elemtype"(%[[C]]) : (!migraphx.shaped<2x3xi32, 6x2x1>) -> !migraphx.shaped<2x3xf32, 6x2x1>
// CHECK:      return %[[CLONE]]
func.func @clone_elemtype_only() -> !migraphx.shaped<2x3xf32, 6x2x1> {
  %c = "migraphx.literal"() {value = dense<1> : tensor<2x3xi32>} : () -> !migraphx.shaped<2x3xi32, 6x2x1>
  %clone = "test.clone_with_elemtype"(%c) {element_type = f32} : (!migraphx.shaped<2x3xi32, 6x2x1>) -> !migraphx.shaped<2x3xf32, 6x2x1>
  return %clone : !migraphx.shaped<2x3xf32, 6x2x1>
}

//===----------------------------------------------------------------------===//
// Test: cloneWith - shape, strides, and element type
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @clone_all
// CHECK:      %[[C:.*]] = "migraphx.literal"() {value = dense<1> : tensor<2x3xi32>} : () -> !migraphx.shaped<2x3xi32, 6x2x1>
// CHECK:      %[[CLONE:.*]] = "test.clone_with_all"(%[[C]]) : (!migraphx.shaped<2x3xi32, 6x2x1>) -> !migraphx.shaped<4x5xf16, 20x4x1>
// CHECK:      return %[[CLONE]]
func.func @clone_all() -> !migraphx.shaped<4x5xf16, 20x4x1> {
  %c = "migraphx.literal"() {value = dense<1> : tensor<2x3xi32>} : () -> !migraphx.shaped<2x3xi32, 6x2x1>
  %clone = "test.clone_with_all"(%c) {shape = [4, 5], strides = [20, 4, 1], element_type = f16} : (!migraphx.shaped<2x3xi32, 6x2x1>) -> !migraphx.shaped<4x5xf16, 20x4x1>
  return %clone : !migraphx.shaped<4x5xf16, 20x4x1>
}

//===----------------------------------------------------------------------===//
// Test: cloneWith - no arguments (should return identical type)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @clone_noop
// CHECK:      %[[C:.*]] = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
// CHECK:      %[[CLONE:.*]] = "test.clone_with_noop"(%[[C]]) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> !migraphx.shaped<2x3xf32, 6x2x1>
// CHECK:      return %[[CLONE]]
func.func @clone_noop() -> !migraphx.shaped<2x3xf32, 6x2x1> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
  %clone = "test.clone_with_noop"(%c) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> !migraphx.shaped<2x3xf32, 6x2x1>
  return %clone : !migraphx.shaped<2x3xf32, 6x2x1>
}

//===----------------------------------------------------------------------===//
// Test: cloneWith - dynamic shape and stride
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @clone_dynamic
// CHECK:      %[[C:.*]] = "migraphx.literal"() {value = dense<1.0> : tensor<?x?xf32>} : () -> !migraphx.shaped<?x?xf32, ?x?>
// CHECK:      %[[CLONE:.*]] = "test.clone_with_shape_strides"(%[[C]]) : (!migraphx.shaped<?x?xf32, ?x?>) -> !migraphx.shaped<4x?xf32, 20x?>
// CHECK:      return %[[CLONE]]
func.func @clone_dynamic() -> !migraphx.shaped<4x?xf32, 20x?> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<?x?xf32>} : () -> !migraphx.shaped<?x?xf32, ?x?>
  %clone = "test.clone_with_shape_strides"(%c) {shape = [4, -1], strides = [20, -1]} : (!migraphx.shaped<?x?xf32, ?x?>) -> !migraphx.shaped<4x?xf32, 20x?>
  return %clone : !migraphx.shaped<4x?xf32, 20x?>
}