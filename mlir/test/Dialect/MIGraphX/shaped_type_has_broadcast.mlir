// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// No broadcast: all strides nonzero
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @no_broadcast
// CHECK:      %[[B:.*]] = "test.has_broadcast"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> i1
// CHECK:      return %[[B]]
func.func @no_broadcast() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
  %b = "test.has_broadcast"(%c) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> i1
  return %b : i1
}

//===----------------------------------------------------------------------===//
// Broadcast: one stride is zero
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_one
// CHECK:      %[[B:.*]] = "test.has_broadcast"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 0x2x1>) -> i1
// CHECK:      return %[[B]]
func.func @broadcast_one() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 0x2x1>
  %b = "test.has_broadcast"(%c) : (!migraphx.shaped<2x3xf32, 0x2x1>) -> i1
  return %b : i1
}

//===----------------------------------------------------------------------===//
// Broadcast: multiple strides are zero
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_multi
// CHECK:      %[[B:.*]] = "test.has_broadcast"(%[[C:.*]]) : (!migraphx.shaped<2x3x4xf32, 0x2x0>) -> i1
// CHECK:      return %[[B]]
func.func @broadcast_multi() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3x4xf32>} : () -> !migraphx.shaped<2x3x4xf32, 0x2x0>
  %b = "test.has_broadcast"(%c) : (!migraphx.shaped<2x3x4xf32, 0x2x0>) -> i1
  return %b : i1
}

//===----------------------------------------------------------------------===//
// No broadcast: dynamic stride, but not zero
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @dynamic_stride_no_broadcast
// CHECK:      %[[B:.*]] = "test.has_broadcast"(%[[C:.*]]) : (!migraphx.shaped<?x3xf32, ?x2x1>) -> i1
// CHECK:      return %[[B]]
func.func @dynamic_stride_no_broadcast() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<?x3xf32>} : () -> !migraphx.shaped<?x3xf32, ?x2x1>
  %b = "test.has_broadcast"(%c) : (!migraphx.shaped<?x3xf32, ?x2x1>) -> i1
  return %b : i1
}

//===----------------------------------------------------------------------===//
// Broadcast: dynamic stride, one is zero
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @dynamic_stride_broadcast
// CHECK:      %[[B:.*]] = "test.has_broadcast"(%[[C:.*]]) : (!migraphx.shaped<?x3xf32, 0x2x?>) -> i1
// CHECK:      return %[[B]]
func.func @dynamic_stride_broadcast() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<?x3xf32>} : () -> !migraphx.shaped<?x3xf32, 0x2x?>
  %b = "test.has_broadcast"(%c) : (!migraphx.shaped<?x3xf32, 0x2x?>) -> i1
  return %b : i1
}

//===----------------------------------------------------------------------===//
// Scalar: no strides, so no broadcast
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @scalar
// CHECK:      %[[B:.*]] = "test.has_broadcast"(%[[C:.*]]) : (!migraphx.shaped<f32>) -> i1
// CHECK:      return %[[B]]
func.func @scalar() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<f32>} : () -> !migraphx.shaped<f32>
  %b = "test.has_broadcast"(%c) : (!migraphx.shaped<f32>) -> i1
  return %b : i1
}