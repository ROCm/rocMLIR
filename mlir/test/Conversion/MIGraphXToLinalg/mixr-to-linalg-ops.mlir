// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg %s -verify-diagnostics -o | FileCheck %s

// CHECK-LABEL: func_sub
// CHECK: linalg.sub 
func.func @func_sub(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.sub %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_mul
// CHECK: linalg.mul 
func.func @func_mul(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.mul %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_div
// CHECK: linalg.div 
func.func @func_div(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.div %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_power
// CHECK: linalg.pow 
func.func @func_power(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.pow %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_abs
// CHECK: linalg.pow 
func.func @func_abs(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.pow %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_ceil
// CHECK: linalg.ceil 
func.func @func_ceil(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.ceil %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_exp
// CHECK: linalg.exp 
func.func @func_exp(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.exp %arg0 : <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_floor
// CHECK: linalg.floor 
func.func @func_floor(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.floor %arg0 : <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_log
// CHECK: linalg.log 
func.func @func_log(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.log %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_neg
// CHECK: linalg.neg 
func.func @func_neg(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.neg %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_sqrt
// CHECK: linalg.sqrt 
func.func @func_sqrt(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.sqrt %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_tanh
// CHECK: linalg.tanh 
func.func @func_tanh(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.tanh %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_recip
// CHECK: linalg.recip 
func.func @func_recip(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.recip %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}
