// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Scalar (no shape, no stride)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @scalar
// CHECK:      -> !migraphx.shaped<f32>
func.func @scalar() -> !migraphx.shaped<f32> {
  %0 = "migraphx.literal"() {value = dense<1.0> : tensor<f32>} : () -> !migraphx.shaped<f32>
  return %0 : !migraphx.shaped<f32>
}

//===----------------------------------------------------------------------===//
// Static shape, static stride
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @static_shape_stride
// CHECK:      -> !migraphx.shaped<2x3xf32, 6x2x1>
func.func @static_shape_stride() -> !migraphx.shaped<2x3xf32, 6x2x1> {
  %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
  return %0 : !migraphx.shaped<2x3xf32, 6x2x1>
}

//===----------------------------------------------------------------------===//
// Dynamic shape, static stride
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @dynamic_shape
// CHECK:      -> !migraphx.shaped<?x3xf32, 6x2x1>
func.func @dynamic_shape() -> !migraphx.shaped<?x3xf32, 6x2x1> {
  %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<?x3xf32, 6x2x1>
  return %0 : !migraphx.shaped<?x3xf32, 6x2x1>
}

//===----------------------------------------------------------------------===//
// Static shape, dynamic stride
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @dynamic_stride
// CHECK:      -> !migraphx.shaped<2x3xf32, ?x2x?>
func.func @dynamic_stride() -> !migraphx.shaped<2x3xf32, ?x2x?> {
  %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, ?x2x?>
  return %0 : !migraphx.shaped<2x3xf32, ?x2x?>
}

//===----------------------------------------------------------------------===//
// Dynamic shape, dynamic stride
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @dynamic_shape_stride
// CHECK:      -> !migraphx.shaped<?x?xf32, ?x?>
func.func @dynamic_shape_stride() -> !migraphx.shaped<?x?xf32, ?x?> {
  %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<?x?xf32, ?x?>
  return %0 : !migraphx.shaped<?x?xf32, ?x?>
}

//===----------------------------------------------------------------------===//
// Integer element type
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @int_type
// CHECK:      -> !migraphx.shaped<2x2xi32, 2x1>
func.func @int_type() -> !migraphx.shaped<2x2xi32, 2x1> {
  %0 = "migraphx.literal"() {value = dense<1> : tensor<2x2xi32>} : () -> !migraphx.shaped<2x2xi32, 2x1>
  return %0 : !migraphx.shaped<2x2xi32, 2x1>
}

//===----------------------------------------------------------------------===//
// Bool element type
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @bool_type
// CHECK:      -> !migraphx.shaped<4xi1, 1>
func.func @bool_type() -> !migraphx.shaped<4xi1, 1> {
  %0 = "migraphx.literal"() {value = dense<true> : tensor<4xi1>} : () -> !migraphx.shaped<4xi1, 1>
  return %0 : !migraphx.shaped<4xi1, 1>
}

//===----------------------------------------------------------------------===//
// One dimension
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @one_dim
// CHECK:      -> !migraphx.shaped<5xf32, 1>
func.func @one_dim() -> !migraphx.shaped<5xf32, 1> {
  %0 = "migraphx.literal"() {value = dense<1.0> : tensor<5xf32>} : () -> !migraphx.shaped<5xf32, 1>
  return %0 : !migraphx.shaped<5xf32, 1>
}

//===----------------------------------------------------------------------===//
// Zero stride (broadcast)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @broadcast_stride
// CHECK:      -> !migraphx.shaped<2x3xf32, 0x2x1>
func.func @broadcast_stride() -> !migraphx.shaped<2x3xf32, 0x2x1> {
  %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 0x2x1>
  return %0 : !migraphx.shaped<2x3xf32, 0x2x1>
}

//===----------------------------------------------------------------------===//
// All dynamic shape and stride
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @all_dynamic
// CHECK:      -> !migraphx.shaped<?x?x?xf32, ?x?x?>
func.func @all_dynamic() -> !migraphx.shaped<?x?x?xf32, ?x?x?> {
  %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3x4xf32>} : () -> !migraphx.shaped<?x?x?xf32, ?x?x?>
  return %0