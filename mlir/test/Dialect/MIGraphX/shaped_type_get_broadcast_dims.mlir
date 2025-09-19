// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// No broadcast: all strides nonzero, expect []
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @no_broadcast
// CHECK:      %[[DIMS:.*]] = "test.get_broadcast_dims"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> tensor<0xi32>
// CHECK:      return %[[DIMS]]
func.func @no_broadcast() -> tensor<0xi32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
  %dims = "test.get_broadcast_dims"(%c) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> tensor<0xi32>
  return %dims : tensor<0xi32>
}

//===----------------------------------------------------------------------===//
// Broadcast: stride 0 at dim 0, expect [0]
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_dim0
// CHECK:      %[[DIMS:.*]] = "test.get_broadcast_dims"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 0x2x1>) -> tensor<1xi32>
// CHECK:      return %[[DIMS]]
func.func @broadcast_dim0() -> tensor<1xi32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 0x2x1>
  %dims = "test.get_broadcast_dims"(%c) : (!migraphx.shaped<2x3xf32, 0x2x1>) -> tensor<1xi32>
  return %dims : tensor<1xi32>
}

//===----------------------------------------------------------------------===//
// Broadcast: stride 0 at dim 1, expect [1]
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_dim1
// CHECK:      %[[DIMS:.*]] = "test.get_broadcast_dims"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 6x0x1>) -> tensor<1xi32>
// CHECK:      return %[[DIMS]]
func.func @broadcast_dim1() -> tensor<1xi32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x0x1>
  %dims = "test.get_broadcast_dims"(%c) : (!migraphx.shaped<2x3xf32, 6x0x1>) -> tensor<1xi32>
  return %dims : tensor<1xi32>
}

//===----------------------------------------------------------------------===//
// Broadcast: stride 0 at dim 2, expect [2]
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_dim2
// CHECK:      %[[DIMS:.*]] = "test.get_broadcast_dims"(%[[C:.*]]) : (!migraphx.shaped<2x3x4xf32, 6x2x0>) -> tensor<1xi32>
// CHECK:      return %[[DIMS]]
func.func @broadcast_dim2() -> tensor<1xi32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3x4xf32>} : () -> !migraphx.shaped<2x3x4xf32, 6x2x0>
  %dims = "test.get_broadcast_dims"(%c) : (!migraphx.shaped<2x3x4xf32, 6x2x0>) -> tensor<1xi32>
  return %dims : tensor<1xi32>
}

//===----------------------------------------------------------------------===//
// Broadcast: stride 0 at dim 0 and 2, expect [0, 2]
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_dim0_2
// CHECK:      %[[DIMS:.*]] = "test.get_broadcast_dims"(%[[C:.*]]) : (!migraphx.shaped<2x3x4xf32, 0x2x0>) -> tensor<2xi32>
// CHECK:      return %[[DIMS]]
func.func @broadcast_dim0_2() -> tensor<2xi32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3x4xf32>} : () -> !migraphx.shaped<2x3x4xf32, 0x2x0>
  %dims = "test.get_broadcast_dims"(%c) : (!migraphx.shaped<2x3x4xf32, 0x2x0>) -> tensor<2xi32>
  return %dims : tensor<2xi32>
}

//===----------------------------------------------------------------------===//
// Broadcast: all strides 0, expect [0, 1, 2]
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_all
// CHECK:      %[[DIMS:.*]] = "test.get_broadcast_dims"(%[[C:.*]]) : (!migraphx.shaped<2x3x4xf32, 0x0x0>) -> tensor<3xi32>
// CHECK:      return %[[DIMS]]
func.func @broadcast_all() -> tensor<3xi32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3x4xf32>} : () -> !migraphx.shaped<2x3x4xf32, 0x0x0>
  %dims = "test.get_broadcast_dims"(%c) : (!migraphx.shaped<2x3x4xf32, 0x0x0>) -> tensor<3xi32>
  return %dims : tensor<3xi32>
}

//===----------------------------------------------------------------------===//
// Scalar: no strides, expect []
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @scalar
// CHECK:      %[[DIMS:.*]] = "test.get_broadcast_dims"(%[[C:.*]]) : (!migraphx.shaped<f32>) -> tensor<0xi32>
// CHECK:      return %[[DIMS]]
func.func @scalar() -> tensor<0xi32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<f32>} : () -> !migraphx.shaped<f32>
  %dims = "test.get_broadcast_dims"(%c) : (!migraphx.shaped<f32>) -> tensor<0xi32>
  return %dims : tensor<0xi32>
}