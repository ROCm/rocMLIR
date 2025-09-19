// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Standard contiguous layout: strides are descending, last stride is 1
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @standard_contiguous
// CHECK:      %[[STD:.*]] = "test.is_standard"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> i1
// CHECK:      return %[[STD]]
func.func @standard_contiguous() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
  %std = "test.is_standard"(%c) : (!migraphx.shaped<2x3xf32, 6x2x1>) -> i1
  return %std : i1
}

//===----------------------------------------------------------------------===//
// Standard: single-dim, stride 1
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @standard_single_dim
// CHECK:      %[[STD:.*]] = "test.is_standard"(%[[C:.*]]) : (!migraphx.shaped<5xf32, 1>) -> i1
// CHECK:      return %[[STD]]
func.func @standard_single_dim() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<5xf32>} : () -> !migraphx.shaped<5xf32, 1>
  %std = "test.is_standard"(%c) : (!migraphx.shaped<5xf32, 1>) -> i1
  return %std : i1
}

//===----------------------------------------------------------------------===//
// Standard: scalar (no stride)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @standard_scalar
// CHECK:      %[[STD:.*]] = "test.is_standard"(%[[C:.*]]) : (!migraphx.shaped<f32>) -> i1
// CHECK:      return %[[STD]]
func.func @standard_scalar() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<f32>} : () -> !migraphx.shaped<f32>
  %std = "test.is_standard"(%c) : (!migraphx.shaped<f32>) -> i1
  return %std : i1
}

//===----------------------------------------------------------------------===//
// Standard: single-dim, stride 0 and shape 1 (broadcasted scalar)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @standard_broadcast_scalar
// CHECK:      %[[STD:.*]] = "test.is_standard"(%[[C:.*]]) : (!migraphx.shaped<1xf32, 0>) -> i1
// CHECK:      return %[[STD]]
func.func @standard_broadcast_scalar() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<1xf32>} : () -> !migraphx.shaped<1xf32, 0>
  %std = "test.is_standard"(%c) : (!migraphx.shaped<1xf32, 0>) -> i1
  return %std : i1
}

//===----------------------------------------------------------------------===//
// Non-standard: strides not descending
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @nonstandard_strides
// CHECK:      %[[STD:.*]] = "test.is_standard"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 2x6x1>) -> i1
// CHECK:      return %[[STD]]
func.func @nonstandard_strides() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 2x6x1>
  %std = "test.is_standard"(%c) : (!migraphx.shaped<2x3xf32, 2x6x1>) -> i1
  return %std : i1
}

//===----------------------------------------------------------------------===//
// Non-standard: missing stride 1
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @nonstandard_no_stride1
// CHECK:      %[[STD:.*]] = "test.is_standard"(%[[C:.*]]) : (!migraphx.shaped<2x3xf32, 12x4x2>) -> i1
// CHECK:      return %[[STD]]
func.func @nonstandard_no_stride1() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 12x4x2>
  %std = "test.is_standard"(%c) : (!migraphx.shaped<2x3xf32, 12x4x2>) -> i1
  return %std : i1
}

//===----------------------------------------------------------------------===//
// Standard: dynamic dims and strides, but sorted and contains 1
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @standard_dynamic
// CHECK:      %[[STD:.*]] = "test.is_standard"(%[[C:.*]]) : (!migraphx.shaped<?x?xf32, ?x1>) -> i1
// CHECK:      return %[[STD]]
func.func @standard_dynamic() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<?x?xf32>} : () -> !migraphx.shaped<?x?xf32, ?x1>
  %std = "test.is_standard"(%c) : (!migraphx.shaped<?x?xf32, ?x1>) -> i1
  return %std : i1
}

//===----------------------------------------------------------------------===//
// Non-standard: dynamic strides, not sorted
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @nonstandard_dynamic
// CHECK:      %[[STD:.*]] = "test.is_standard"(%[[C:.*]]) : (!migraphx.shaped<?x?xf32, 2x?>) -> i1
// CHECK:      return %[[STD]]
func.func @nonstandard_dynamic() -> i1 {
  %c = "migraphx.literal"() {value = dense<1.0> : tensor<?x?xf32>} : () -> !migraphx.shaped<?x?xf32, 2x?>
  %std = "test.is_standard"(%c) : (!migraphx.shaped<?x?xf32, 2x?>) -> i1
  return %std : i1
}