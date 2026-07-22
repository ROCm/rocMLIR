// RUN: rocmlir-opt --migraphx-to-tosa --mlir-print-op-generic -verify-diagnostics %s | FileCheck %s --check-prefix=TOSA
// RUN: rocmlir-opt -pass-pipeline="builtin.module(func.func(migraphx-to-tosa),func.func(rocmlir-custom-tosa-to-linalg),func.func(tosa-to-linalg-named),func.func(tosa-to-linalg))" -verify-diagnostics %s | FileCheck %s --check-prefixes=LINALG,UNSIGNED

// IEEE maximum propagates a NaN from either operand, orders +0 above -0, and
// handles infinities using their normal ordering. Check both the explicit TOSA
// contract and the scalar operation selected by TOSA-to-Linalg.

// TOSA: sym_name = "max_f32"
// TOSA: "tosa.maximum"
// TOSA-SAME: nan_mode
// TOSA-SAME: PROPAGATE
// TOSA-NOT: IGNORE

// LINALG-LABEL: func.func @max_f32
// LINALG: arith.maximumf
// LINALG-NOT: arith.maxnumf
func.func @max_f32(%arg0: !migraphx.shaped<2x4xf32, 4x1>, %arg1: !migraphx.shaped<2x4xf32, 0x1>) -> !migraphx.shaped<2x4xf32, 4x1> {
  %0 = migraphx.max %arg0, %arg1 : <2x4xf32, 4x1>, <2x4xf32, 0x1> -> <2x4xf32, 4x1>
  return %0 : !migraphx.shaped<2x4xf32, 4x1>
}

// TOSA: sym_name = "max_unit_zero_stride"
// TOSA: "tosa.maximum"
// TOSA-SAME: nan_mode
// TOSA-SAME: PROPAGATE

// LINALG-LABEL: func.func @max_unit_zero_stride
// LINALG: arith.maximumf
func.func @max_unit_zero_stride(%arg0: !migraphx.shaped<1xf32, 0>, %arg1: !migraphx.shaped<1xf32, 0>) -> !migraphx.shaped<1xf32, 0> {
  %0 = migraphx.max %arg0, %arg1 : <1xf32, 0>, <1xf32, 0> -> <1xf32, 0>
  return %0 : !migraphx.shaped<1xf32, 0>
}

// TOSA: sym_name = "max_ui32_broadcast"
// TOSA: "tosa.custom"
// TOSA-SAME: operator_name = "unsigned_max"

// UNSIGNED-LABEL: func.func @max_ui32_broadcast
// UNSIGNED-NOT: arith.maxsi
// UNSIGNED: %[[BROADCAST:.+]] = linalg.generic
// UNSIGNED-SAME: ins(%{{.+}}, %{{.+}} : tensor<2x4xi32>, tensor<1x4xi32>)
// UNSIGNED: linalg.generic
// UNSIGNED-SAME: ins(%{{.+}}, %[[BROADCAST]] : tensor<2x4xi32>, tensor<2x4xi32>)
// UNSIGNED: arith.maxui
// UNSIGNED-NOT: arith.maxsi
func.func @max_ui32_broadcast(%arg0: !migraphx.shaped<2x4xui32, 4x1>, %arg1: !migraphx.shaped<2x4xui32, 0x1>) -> !migraphx.shaped<2x4xui32, 4x1> {
  %0 = migraphx.max %arg0, %arg1 : <2x4xui32, 4x1>, <2x4xui32, 0x1> -> <2x4xui32, 4x1>
  return %0 : !migraphx.shaped<2x4xui32, 4x1>
}
