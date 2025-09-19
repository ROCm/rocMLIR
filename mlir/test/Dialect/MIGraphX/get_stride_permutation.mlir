// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Standard contiguous: strides descending, permutation is [0, 1, 2]
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @standard_contiguous
// CHECK:      %[[P:.*]] = "test.get_stride_permutation"(%[[C:.*]]) : (!migraphx.shaped<2x3x4xf32, 12x4x1>) -> tensor<3xi64>
// CHECK:      %[[VAL:.*]] = arith.constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK:      %[[EQ:.*]] = arith.cmpi eq, %[[P]], %[[VAL]]
// CHECK:      return %[[EQ]]
func.func @standard_contiguous() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3x4xf32>} : () -> !migraphx.shaped<2x3x4xf32, 12x4x1>
  %p = "test.get_stride_permutation"(%c) : (!migraphx.shaped<2x3x4xf32, 12x4x1>) -> tensor<3xi64>
  %val = arith.constant dense<[0, 1, 2]> : tensor<3xi64>
  %eq = arith.cmpi eq, %p, %val : tensor<3xi64>
  return %eq : i1
}

//===----------------------------------------------------------------------===//
// Reverse order: strides ascending, permutation is [2, 1, 0]
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @reverse_order
// CHECK:      %[[P:.*]] = "test.get_stride_permutation"(%[[C:.*]]) : (!migraphx.shaped<2x3x4xf32, 1x4x12>) -> tensor<3xi64>
// CHECK:      %[[VAL:.*]] = arith.constant dense<[2, 1, 0]> : tensor<3xi64>
// CHECK:      %[[EQ:.*]] = arith.cmpi eq, %[[P]], %[[VAL]]
// CHECK:      return %[[EQ]]
func.func @reverse_order() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3x4xf32>} : () -> !migraphx.shaped<2x3x4xf32, 1x4x12>
  %p = "test.get_stride_permutation"(%c) : (!migraphx.shaped<2x3x4xf32, 1x4x12>) -> tensor<3xi64>
  %val = arith.constant dense<[2, 1, 0]> : tensor<3xi64>
  %eq = arith.cmpi eq, %p, %val : tensor<3xi64>
  return %eq : i1
}

//===----------------------------------------------------------------------===//
// Mixed order: strides [4, 1, 12], permutation is [2, 0, 1]
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @mixed_order
// CHECK:      %[[P:.*]] = "test.get_stride_permutation"(%[[C:.*]]) : (!migraphx.shaped<2x3x4xf32, 4x1x12>) -> tensor<3xi64>
// CHECK:      %[[VAL:.*]] = arith.constant dense<[2, 0, 1]> : tensor<3xi64>
// CHECK:      %[[EQ:.*]] = arith.cmpi eq, %[[P]], %[[VAL]]
// CHECK:      return %[[EQ]]
func.func @mixed_order() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3x4xf32>} : () -> !migraphx.shaped<2x3x4xf32, 4x1x12>
  %p = "test.get_stride_permutation"(%c) : (!migraphx.shaped<2x3x4xf32, 4x1x12>) -> tensor<3xi64>
  %val = arith.constant dense<[2, 0, 1]> : tensor<3xi64>
  %eq = arith.cmpi eq, %p, %val : tensor<3xi64>
  return %eq : i1
}

//===----------------------------------------------------------------------===//
// Broadcast: stride 0 at dim 1, permutation is [0, 2, 1]
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_dim1
// CHECK:      %[[P:.*]] = "test.get_stride_permutation"(%[[C:.*]]) : (!migraphx.shaped<2x3x4xf32, 12x0x1>) -> tensor<3xi64>
// CHECK:      %[[VAL:.*]] = arith.constant dense<[0, 2, 1]> : tensor<3xi64>
// CHECK:      %[[EQ:.*]] = arith.cmpi eq, %[[P]], %[[VAL]]
// CHECK:      return %[[EQ]]
func.func @broadcast_dim1() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3x4xf32>} : () -> !migraphx.shaped<2x3x4xf32, 12x0x1>
  %p = "test.get_stride_permutation"(%c) : (!migraphx.shaped<2x3x4xf32, 12x0x1>) -> tensor<3xi64>
  %val = arith.constant dense<[0, 2, 1]> : tensor<3xi64>
  %eq = arith.cmpi eq, %p, %val : tensor<3xi64>
  return %eq : i1
}

//===----------------------------------------------------------------------===//
// All broadcast: all strides 0, permutation is [0, 1, 2]
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @all_broadcast
// CHECK:      %[[P:.*]] = "test.get_stride_permutation"(%[[C:.*]]) : (!migraphx.shaped<2x3x4xf32, 0x0x0>) -> tensor<3xi64>
// CHECK:      %[[VAL:.*]] = arith.constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK:      %[[EQ:.*]] = arith.cmpi eq, %[[P]], %[[VAL]]
// CHECK:      return %[[EQ]]
func.func @all_broadcast() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3x4xf32>} : () -> !migraphx.shaped<2x3x4xf32, 0x0x0>
  %p = "test.get_stride_permutation"(%c) : (!migraphx.shaped<2x3x4xf32, 0x0x0>) -> tensor<3xi64>
  %val = arith.constant dense<[0, 1, 2]> : tensor<3xi64>
  %eq = arith.cmpi eq, %p, %val : tensor<3xi64>
  return %eq : i1
}

//===----------------------------------------------------------------------===//
// Scalar: no strides, permutation is []
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @scalar
// CHECK:      %[[P:.*]] = "test.get_stride_permutation"(%[[C:.*]]) : (!migraphx.shaped<f32>) -> tensor<0xi64>
// CHECK:      %[[VAL:.*]] = arith.constant dense<[]> : tensor<0xi64>
// CHECK:      %[[EQ:.*]] = arith.cmpi eq, %[[P]], %[[VAL]]
// CHECK:      return %[[EQ]]
func.func @scalar() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<f32>} : () -> !migraphx.shaped<f32>
  %p = "test.get_stride_permutation"(%c) : (!migraphx.shaped<f32>) -> tensor<0xi64>
  %val = arith.constant dense<[]> : tensor<0xi64>
  %eq = arith.cmpi eq, %p, %val : tensor<0xi64>
  return %eq : i1
}