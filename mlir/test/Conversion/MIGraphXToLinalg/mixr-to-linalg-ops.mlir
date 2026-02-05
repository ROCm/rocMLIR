// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg %s -verify-diagnostics  | FileCheck %s

// CHECK-LABEL: func_sub
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.sub ins(%[[expanded_0]], %[[expanded]] {{.*}})
func.func @func_sub(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.sub %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_mul
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.mul ins(%[[expanded_0]], %[[expanded]] {{.*}})
func.func @func_mul(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.mul %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_div
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.div ins(%[[expanded_0]], %[[expanded]] {{.*}})
func.func @func_div(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.div %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_power
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.powf ins(%[[expanded_0]], %[[expanded]] {{.*}})
func.func @func_power(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.pow %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_abs
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.abs ins(%[[expanded_0]] {{.*}})
func.func @func_abs(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.abs %arg0 : <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_ceil
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.ceil ins(%[[expanded]] {{.*}})
func.func @func_ceil(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.ceil %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_exp
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.exp ins(%[[expanded_0]] {{.*}})
func.func @func_exp(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.exp %arg0 : <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_floor
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.floor ins(%[[expanded_0]] {{.*}})
func.func @func_floor(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.floor %arg0 : <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_log
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.log ins(%[[expanded_0]] {{.*}})
func.func @func_log(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.log %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_neg
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.negf ins(%[[expanded_0]] {{.*}})
func.func @func_neg(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.neg %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_sqrt
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.sqrt ins(%[[expanded_0]] {{.*}})
func.func @func_sqrt(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.sqrt %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_tanh
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.tanh ins(%[[expanded_0]] {{.*}})
func.func @func_tanh(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.tanh %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_recip
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.reciprocal ins(%[[expanded_0]] {{.*}})
func.func @func_recip(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.recip %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}
