// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Standard contiguous layout: should produce same shape as input
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @standard_contiguous
// CHECK:      %[[T:.*]] = "test.as_memory_layout_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> tensor<2x3xf32>
// CHECK:      return %[[T]]
func.func @standard_contiguous() -> tensor<2x3xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
  %t = "test.as_memory_layout_tensor"(%c) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> tensor<2x3xf32>
  return %t : tensor<2x3xf32>
}

//===----------------------------------------------------------------------===//
// Broadcast: stride 0, shape should be 1 in broadcasted dim
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_dim0
// CHECK:      %[[T:.*]] = "test.as_memory_layout_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 0x2x1>) -> tensor<1x3xf32>
// CHECK:      return %[[T]]
func.func @broadcast_dim0() -> tensor<1x3xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 0x2x1>
  %t = "test.as_memory_layout_tensor"(%c) : (!migraphx.shaped<2x3xf32, 0x2x1>) -> tensor<1x3xf32>
  return %t : tensor<1x3xf32>
}

//===----------------------------------------------------------------------===//
// Broadcast: stride 0 at dim 1, shape should be 1 in that dim
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_dim1
// CHECK:      %[[T:.*]] = "test.as_memory_layout_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 6x0x1>) -> tensor<2x1xf32>
// CHECK:      return %[[T]]
func.func @broadcast_dim1() -> tensor<2x1xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x0x1>
  %t = "test.as_memory_layout_tensor"(%c) : (!migraphx.shaped<2x3xf32, 6x0x1>) -> tensor<2x1xf32>
  return %t : tensor<2x1xf32>
}

//===----------------------------------------------------------------------===//
// Broadcast: stride 0 at dim 0 and 2, shape should be 1 in those dims
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_dim0_2
// CHECK:      %[[T:.*]] = "test.as_memory_layout_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x3x4xf32, 0x2x0>) -> tensor<1x3x1xf32>
// CHECK:      return %[[T]]
func.func @broadcast_dim0_2() -> tensor<1x3x1xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3x4xf32>} : () -> !migraphx.shaped<2x3x4xf32, 0x2x0>
  %t = "test.as_memory_layout_tensor"(%c) : (!migraphx.shaped<2x3x4xf32, 0x2x0>) -> tensor<1x3x1xf32>
  return %t : tensor<1x3x1xf32>
}

//===----------------------------------------------------------------------===//
// Scalar: should produce tensor<f32>
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @scalar
// CHECK:      %[[T:.*]] = "test.as_memory_layout_tensor"(%[[C:.*]]) : (!migraphx.shaped<f32>) -> tensor<f32>
// CHECK:      return %[[T]]
func.func @scalar() -> tensor<f32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<f32>} : () -> !migraphx.shaped<f32>
  %t = "test.as_memory_layout_tensor"(%c) : (!migraphx.shaped<f32>) -> tensor<f32>
  return %t : tensor<f32>
}

//===----------------------------------------------------------------------===//
// Non-standard: strides not in standard order, expect error (nullptr)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @nonstandard_strides
// CHECK:      %[[T:.*]] = "test.as_memory_layout_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 2x6x1>) -> !nulltype
// CHECK:      return %[[T]]
func.func @nonstandard_strides() -> !nulltype {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 2x6x1>
  %t = "test.as_memory_layout_tensor"(%c) : (!migraphx.shaped<2x3xf32, 2x6x1>) -> !nulltype
  return %t : !nulltype
}

//===----------------------------------------------------------------------===//
// Non-standard: missing stride 1, expect error (nullptr)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @nonstandard_no_stride1
// CHECK:      %[[T:.*]] = "test.as_memory_layout_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 12x4x2>) -> !nulltype
// CHECK:      return %[[T]]
func.func @nonstandard_no_stride1() -> !nulltype {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 12x4x2>
  %t = "test.as_memory_layout_tensor"(%c) : (!migraphx.shaped<2x3xf32, 12x4x2>) -> !nulltype
  return %t : !nulltype
}

//===----------------------------------------------------------------------===//
// Integer element type: should produce signless tensor
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @int_type
// CHECK:      %[[T:.*]] = "test.as_memory_layout_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x2xi32, 2x1>) -> tensor<2x2xi32>
// CHECK:      return %[[T]]
func.func @int_type() -> tensor<2x2xi32> {
  %c = "migraphx.literal"() {value = dense<1> : tensor<2x2xi32>} : () -> !migraphx.shaped<2x2xi32, 2x1>
  %t = "test.as_memory_layout_tensor"(%c) : (!migraphx.shaped<2x2xi32, 2x1>) -> tensor<2x2xi32>
  return %t : tensor<2x2xi32>
}

//===----------------------------------------------------------------------===//
// Dynamic shape and stride: should produce tensor with dynamic dims
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @dynamic_shape_stride
// CHECK:      %[[T:.*]] = "test.as_memory_layout_tensor"(%[[C:.*]]) : (!migraphx.shaped<?x?xf32, ?x?>) -> tensor<?x?xf32>
// CHECK:      return %[[T]]
func.func @dynamic_shape_stride() -> tensor<?x?xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<?x?xf32>} : () -> !migraphx.shaped<?x?xf32, ?x?>
  %t = "test.as_memory_layout_tensor"(%c) : (!migraphx.shaped<?x?xf32, ?x?>) -> tensor<?x?xf32>
  return %t : tensor<?x?xf32>
}