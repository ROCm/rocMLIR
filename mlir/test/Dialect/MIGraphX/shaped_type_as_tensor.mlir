// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Scalar: asTensor should produce ranked tensor with no dims
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @scalar
// CHECK:      %[[T:.*]] = "test.as_tensor"(%[[C:.*]]) : (!migraphx.shaped<f32>) -> tensor<f32>
// CHECK:      return %[[T]]
func.func @scalar() -> tensor<f32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<f32>} : () -> !migraphx.shaped<f32>
  %t = "test.as_tensor"(%c) : (!migraphx.shaped<f32>) -> tensor<f32>
  return %t : tensor<f32>
}

//===----------------------------------------------------------------------===//
// Static shape: asTensor should produce ranked tensor with same shape
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @static_shape
// CHECK:      %[[T:.*]] = "test.as_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> tensor<2x3xf32>
// CHECK:      return %[[T]]
func.func @static_shape() -> tensor<2x3xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
  %t = "test.as_tensor"(%c) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> tensor<2x3xf32>
  return %t : tensor<2x3xf32>
}

//===----------------------------------------------------------------------===//
// Dynamic shape: asTensor should produce ranked tensor with dynamic dims
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @dynamic_shape
// CHECK:      %[[T:.*]] = "test.as_tensor"(%[[C:.*]]) : (!migraphx.shaped<?x3xf32, 6x2x1>) -> tensor<?x3xf32>
// CHECK:      return %[[T]]
func.func @dynamic_shape() -> tensor<?x3xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<?x3xf32, 6x2x1>
  %t = "test.as_tensor"(%c) : (!migraphx.shaped<?x3xf32, 6x2x1>) -> tensor<?x3xf32>
  return %t : tensor<?x3xf32>
}

//===----------------------------------------------------------------------===//
// Dynamic rank: asTensor should produce ranked tensor with all dynamic dims
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @all_dynamic
// CHECK:      %[[T:.*]] = "test.as_tensor"(%[[C:.*]]) : (!migraphx.shaped<?x?xf32, ?x?>) -> tensor<?x?xf32>
// CHECK:      return %[[T]]
func.func @all_dynamic() -> tensor<?x?xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<?x?xf32, ?x?>
  %t = "test.as_tensor"(%c) : (!migraphx.shaped<?x?xf32, ?x?>) -> tensor<?x?xf32>
  return %t : tensor<?x?xf32>
}

//===----------------------------------------------------------------------===//
// Integer element type: asTensor should preserve element type
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @int_type
// CHECK:      %[[T:.*]] = "test.as_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x2xi32, 2x1>) -> tensor<2x2xi32>
// CHECK:      return %[[T]]
func.func @int_type() -> tensor<2x2xi32> {
  %c = "migraphx.literal"() {value = dense<1> : tensor<2x2xi32>} : () -> !migraphx.shaped<2x2xi32, 2x1>
  %t = "test.as_tensor"(%c) : (!migraphx.shaped<2x2xi32, 2x1>) -> tensor<2x2xi32>
  return %t : tensor<2x2xi32>
}

//===----------------------------------------------------------------------===//
// Bool element type: asTensor should preserve element type
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @bool_type
// CHECK:      %[[T:.*]] = "test.as_tensor"(%[[C:.*]]) : (!migraphx.shaped<4xi1, 1>) -> tensor<4xi1>
// CHECK:      return %[[T]]
func.func @bool_type() -> tensor<4xi1> {
  %c = "migraphx.literal"() {value = dense<true> : tensor<4xi1>} : () -> !migraphx.shaped<4xi1, 1>
  %t = "test.as_tensor"(%c) : (!migraphx.shaped<4xi1, 1>) -> tensor<4xi1>
  return %t : tensor<4xi1>
}

//===----------------------------------------------------------------------===//
// One dimension: asTensor should produce tensor with one dim
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @one_dim
// CHECK:      %[[T:.*]] = "test.as_tensor"(%[[C:.*]]) : (!migraphx.shaped<5xf32, 1>) -> tensor<5xf32>
// CHECK:      return %[[T]]
func.func @one_dim() -> tensor<5xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<5xf32>} : () -> !migraphx.shaped<5xf32, 1>
  %t = "test.as_tensor"(%c) : (!migraphx.shaped<5xf32, 1>) -> tensor<5xf32>
  return %t : tensor<5xf32>
}