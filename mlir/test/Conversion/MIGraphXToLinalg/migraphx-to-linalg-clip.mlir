// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @func_clip(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}}, %[[arg2:.*]]: tensor{{.*}})
// CHECK-DAG:  %[[expanded:.*]] = tensor.expand_shape %[[arg2]]
// CHECK-DAG:  %[[expanded_0:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:  %[[expanded_1:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:  %[[cst:.*]] = arith.constant
// CHECK-DAG:  %[[cst_2:.*]] = arith.constant
// CHECK-DAG:  %[[zero:.*]] = linalg.max ins(%[[expanded_1]], %[[expanded_0]] : {{.*}}) outs(%[[cst]] : {{.*}})
// CHECK-DAG:  %[[one:.*]] = linalg.min ins(%[[zero]], %[[expanded]] : {{.*}}) outs(%cst_2 : {{.*}})
// CHECK-DAG:  %[[collapsed:.*]] = tensor.collapse_shape %[[one]]
// CHECK-DAG:  return %[[collapsed]]
func.func @func_clip(%x: !migraphx.shaped<1x1xf32, 1x1>, %min: !migraphx.shaped<1x1xf32, 1x1>, %max: !migraphx.shaped<1x1xf32, 1x1>) -> !migraphx.shaped<1x1xf32, 1x1> {
  %clipped = migraphx.clip %x, %min, %max: <1x1xf32, 1x1>, <1x1xf32, 1x1>, <1x1xf32, 1x1> -> <1x1xf32, 1x1>
  func.return %clipped : !migraphx.shaped<1x1xf32, 1x1>
}

// testcase from mixr-to-tosa-ops.mlir

// CHECK-LABEL: @clip_i32(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}}, %[[arg2:.*]]: tensor{{.*}})
// CHECK-DAG:  %[[expanded:.*]] = tensor.expand_shape %[[arg2]]
// CHECK-DAG:  %[[expanded_0:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:  %[[expanded_1:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:  %[[cst:.*]] = arith.constant
// CHECK-DAG:  %[[cst_2:.*]] = arith.constant
// CHECK-DAG:  %[[zero:.*]] = linalg.max ins(%[[expanded_1]], %[[expanded_0]] : {{.*}}) outs(%[[cst]] : {{.*}})
// CHECK-DAG:  %[[one:.*]] = linalg.min ins(%[[zero]], %[[expanded]] : {{.*}}) outs(%cst_2 : {{.*}})
// CHECK-DAG:  %[[collapsed:.*]] = tensor.collapse_shape %[[one]]
// CHECK-DAG:  return %[[collapsed]]
func.func @clip_i32(%arg0: !migraphx.shaped<64x64xi32, 64x1>, %arg1: !migraphx.shaped<64x64xi32, 64x1>, %arg2: !migraphx.shaped<64x64xi32, 64x1>) -> !migraphx.shaped<64x64xi32, 64x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
  %0 = migraphx.clip %arg0, %arg1, %arg2 : <64x64xi32, 64x1>, <64x64xi32, 64x1>, <64x64xi32, 64x1> -> <64x64xi32, 64x1>
  return %0 : !migraphx.shaped<64x64xi32, 64x1>
}
