// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Standard contiguous layout: should produce flat tensor with total elements
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @standard_contiguous
// CHECK:      %[[T:.*]] = "test.as_flat_memory_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> tensor<6xf32>
// CHECK:      return %[[T]]
func.func @standard_contiguous() -> tensor<6xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
  %t = "test.as_flat_memory_tensor"(%c) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> tensor<6xf32>
  return %t : tensor<6xf32>
}

//===----------------------------------------------------------------------===//
// Broadcast: stride 0, shape should be 1 in broadcasted dim, flat = product of non-broadcast dims
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_dim0
// CHECK:      %[[T:.*]] = "test.as_flat_memory_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 0x2x1>) -> tensor<3xf32>
// CHECK:      return %[[T]]
func.func @broadcast_dim0() -> tensor<3xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 0x2x1>
  %t = "test.as_flat_memory_tensor"(%c) : (!migraphx.shaped<2x3xf32, 0x2x1>) -> tensor<3xf32>
  return %t : tensor<3xf32>
}

//===----------------------------------------------------------------------===//
// Broadcast: stride 0 at dim 1, flat = product of non-broadcast dims
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_dim1
// CHECK:      %[[T:.*]] = "test.as_flat_memory_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 6x0x1>) -> tensor<2xf32>
// CHECK:      return %[[T]]
func.func @broadcast_dim1() -> tensor<2xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x0x1>
  %t = "test.as_flat_memory_tensor"(%c) : (!migraphx.shaped<2x3xf32, 6x0x1>) -> tensor<2xf32>
  return %t : tensor<2xf32>
}

//===----------------------------------------------------------------------===//
// Broadcast: stride 0 at dim 0 and 2, flat = product of non-broadcast dims
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_dim0_2
// CHECK:      %[[T:.*]] = "test.as_flat_memory_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x3x4xf32, 0x2x0>) -> tensor<3xf32>
// CHECK:      return %[[T]]
func.func @broadcast_dim0_2() -> tensor<3xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3x4xf32>} : () -> !migraphx.shaped<2x3x4xf32, 0x2x0>
  %t = "test.as_flat_memory_tensor"(%c) : (!migraphx.shaped<2x3x4xf32, 0x2x0>) -> tensor<3xf32>
  return %t : tensor<3xf32>
}

//===----------------------------------------------------------------------===//
// Scalar: should produce tensor<1xf32>
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @scalar
// CHECK:      %[[T:.*]] = "test.as_flat_memory_tensor"(%[[C:.*]]) : (!migraphx.shaped<f32>) -> tensor<1xf32>
// CHECK:      return %[[T]]
func.func @scalar() -> tensor<1xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<f32>} : () -> !migraphx.shaped<f32>
  %t = "test.as_flat_memory_tensor"(%c) : (!migraphx.shaped<f32>) -> tensor<1xf32>
  return %t : tensor<1xf32>
}

//===----------------------------------------------------------------------===//
// Non-standard: strides not in standard order, expect error (nullptr)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @nonstandard_strides
// CHECK:      %[[T:.*]] = "test.as_flat_memory_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 2x6x1>) -> !nulltype
// CHECK:      return %[[T]]
func.func @nonstandard_strides() -> !nulltype {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 2x6x1>
  %t = "test.as_flat_memory_tensor"(%c) : (!migraphx.shaped<2x3xf32, 2x6x1>) -> !nulltype
  return %t : !nulltype
}

//===----------------------------------------------------------------------===//
// Non-standard: missing stride 1, expect error (nullptr)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @nonstandard_no_stride1
// CHECK:      %[[T:.*]] = "test.as_flat_memory_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 12x4x2>) -> !nulltype
// CHECK:      return %[[T]]
func.func @nonstandard_no_stride1() -> !nulltype {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 12x4x2>
  %t = "test.as_flat_memory_tensor"(%c) : (!migraphx.shaped<2x3xf32, 12x4x2>) -> !nulltype
  return %t : !nulltype
}

//===----------------------------------------------------------------------===//
// Integer element type: should produce signless tensor
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @int_type
// CHECK:      %[[T:.*]] = "test.as_flat_memory_tensor"(%[[C:.*]]) : (!migraphx.shaped<2x2xi32, 2x1>) -> tensor<4xi32>
// CHECK:      return %[[T]]
func.func @int_type() -> tensor<4xi32> {
  %c = "migraphx.literal"() {value = dense<1> : tensor<2x2xi32>} : () -> !migraphx.shaped<2x2xi32, 2x1>
  %t = "test.as_flat_memory_tensor"(%c) : (!migraphx.shaped<2x2xi32, 2x1>) -> tensor<4xi32>
  return %t : tensor<4xi32>
}

//===----------------------------------------------------------------------===//
// Dynamic shape and stride: should produce tensor with dynamic dim
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @dynamic_shape_stride
// CHECK:      %[[T:.*]] = "test.as_flat_memory_tensor"(%[[C:.*]]) : (!migraphx.shaped<?x?xf32, ?x?>) -> tensor<?xf32>
// CHECK:      return %[[T]]
func.func @dynamic_shape_stride() -> tensor<?xf32> {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<?x?xf32>} : () -> !migraphx.shaped<?x?xf32, ?x?>
  %t = "test.as_flat_memory_tensor"(%c) : (!migraphx.shaped<?x?xf32, ?x?>) -> tensor<?xf32>
  return %t : tensor<?xf32>
}