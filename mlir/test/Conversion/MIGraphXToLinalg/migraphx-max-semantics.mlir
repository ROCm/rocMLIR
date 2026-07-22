// RUN: rocmlir-opt --migraphx-to-linalg --linalg-generalize-named-ops -verify-diagnostics %s | FileCheck %s

// IEEE maximum propagates a NaN from either operand, orders +0 above -0, and
// handles infinities using their normal ordering.

// CHECK-LABEL: func.func @max_f32
// CHECK: arith.maximumf
// CHECK-NOT: arith.maxnumf
func.func @max_f32(%arg0: !migraphx.shaped<2x4xf32, 4x1>, %arg1: !migraphx.shaped<2x4xf32, 0x1>) -> !migraphx.shaped<2x4xf32, 4x1> {
  %0 = migraphx.max %arg0, %arg1 : <2x4xf32, 4x1>, <2x4xf32, 0x1> -> <2x4xf32, 4x1>
  return %0 : !migraphx.shaped<2x4xf32, 4x1>
}

// CHECK-LABEL: func.func @max_ui32_broadcast
// CHECK: tensor.expand_shape
// CHECK: arith.maxui
// CHECK-NOT: arith.maxsi
func.func @max_ui32_broadcast(%arg0: !migraphx.shaped<2x4xui32, 4x1>, %arg1: !migraphx.shaped<2x4xui32, 0x1>) -> !migraphx.shaped<2x4xui32, 4x1> {
  %0 = migraphx.max %arg0, %arg1 : <2x4xui32, 4x1>, <2x4xui32, 0x1> -> <2x4xui32, 4x1>
  return %0 : !migraphx.shaped<2x4xui32, 4x1>
}

// CHECK-LABEL: func.func @max_unit_zero_stride
// CHECK: arith.maximumf
func.func @max_unit_zero_stride(%arg0: !migraphx.shaped<1xf32, 0>, %arg1: !migraphx.shaped<1xf32, 0>) -> !migraphx.shaped<1xf32, 0> {
  %0 = migraphx.max %arg0, %arg1 : <1xf32, 0>, <1xf32, 0> -> <1xf32, 0>
  return %0 : !migraphx.shaped<1xf32, 0>
}
