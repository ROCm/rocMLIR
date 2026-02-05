// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func_relu(
// CHECK-SAME: %[[arg0:.*]]: tensor
// CHECK-DAG:  %[[expanded:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:  %[[cst:.*]] = arith.constant
// CHECK-DAG:  %[[cst_0:.*]] = arith.constant
// CHECK-DAG:  %[[zero:.*]] = linalg.max ins(%[[expanded]], %[[cst]] : {{.*}}) outs(%[[cst_0]] : {{.*}})
// CHECK-DAG:  %[[collapsed:.*]] = tensor.collapse_shape %[[zero]]
// CHECK-DAG:  return %[[collapsed]]
func.func @func_relu(%arg0: !migraphx.shaped<1x1xf32, 1x1>) -> !migraphx.shaped<1x1xf32, 1x1> {
  %arg1 = migraphx.relu %arg0: <1x1xf32, 1x1> -> <1x1xf32, 1x1>
  func.return %arg1: !migraphx.shaped<1x1xf32, 1x1> 
}
