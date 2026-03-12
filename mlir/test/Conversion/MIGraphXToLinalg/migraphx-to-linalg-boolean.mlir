// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @func_greater(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[zero:.*]] = tensor.empty
// CHECK-DAG: %[[one:.*]] = linalg.generic {{.*}} ins(%[[expanded_0]], %[[expanded]] : tensor{{.*}}) outs(%[[zero]] : tensor{{.*}}) {
// CHECK-DAG:    ^bb0(%[[in:.*]]: i32, %[[in_1:.*]]: i32, %[[out:.*]]: i32):
// CHECK-DAG:        %[[two:.*]] = arith.cmpi sgt, %[[in]], %[[in_1]]
// CHECK-DAG:        %[[three:.*]] = arith.extui %[[two]] : i1 to i32
// CHECK-DAG:        linalg.yield %[[three]]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[one]]
// CHECK-DAG: return %[[collapsed]]
func.func @func_greater(%arg0: !migraphx.shaped<1x1x3xi32, 3x3x1>, %arg1: !migraphx.shaped<1x1x3xi32, 3x3x1>) ->  !migraphx.shaped<1x1x3xi32, 3x3x1> attributes {kernel, arch="gfx950"} {
  %result = migraphx.greater %arg0, %arg1: <1x1x3xi32, 3x3x1>, <1x1x3xi32, 3x3x1> -> <1x1x3xi32, 3x3x1>
  func.return %result : !migraphx.shaped<1x1x3xi32, 3x3x1>
}

// CHECK-LABEL: @func_greater_signed(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[zero:.*]] = tensor.empty
// CHECK-DAG: %[[one:.*]] = linalg.generic {{.*}} ins(%[[expanded_0]], %[[expanded]] : tensor{{.*}}) outs(%[[zero]] : tensor{{.*}}) {
// CHECK-DAG:    ^bb0(%[[in:.*]]: {{.*}}, %[[in_1:.*]]: {{.*}}, %[[out:.*]]: {{.*}}):
// CHECK-DAG:        %[[two:.*]] = arith.cmpi sgt, %[[in]], %[[in_1]]
// CHECK-DAG:        %[[three:.*]] = arith.extui %[[two]] : i1 to i32
// CHECK-DAG:        linalg.yield %[[three]]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[one]]
// CHECK-DAG: return %[[collapsed]]
func.func @func_greater_signed(%arg0: !migraphx.shaped<1x1x3xsi32, 3x3x1>, %arg1: !migraphx.shaped<1x1x3xsi32, 3x3x1>) ->  !migraphx.shaped<1x1x3xsi32, 3x3x1> attributes {kernel, arch="gfx950"} {
  %result = migraphx.greater %arg0, %arg1: <1x1x3xsi32, 3x3x1>, <1x1x3xsi32, 3x3x1> -> <1x1x3xsi32, 3x3x1>
  func.return %result : !migraphx.shaped<1x1x3xsi32, 3x3x1>
}

// CHECK-LABEL: @func_greater_float(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[zero:.*]] = tensor.empty
// CHECK-DAG: %[[one:.*]] = linalg.generic {{.*}} ins(%[[expanded_0]], %[[expanded]] : tensor{{.*}}) outs(%[[zero]] : tensor{{.*}}) {
// CHECK-DAG:    ^bb0(%[[in:.*]]: {{.*}}, %[[in_1:.*]]: {{.*}}, %[[out:.*]]: {{.*}}):
// CHECK-DAG:        %[[two:.*]] = arith.cmpf ogt, %[[in]], %[[in_1]]
// CHECK-DAG:        %[[three:.*]] = arith.uitofp %[[two]]
// CHECK-DAG:        linalg.yield %[[three]]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[one]]
// CHECK-DAG: return %[[collapsed]]
func.func @func_greater_float(%arg0: !migraphx.shaped<1x1x3xf32, 3x3x1>, %arg1: !migraphx.shaped<1x1x3xf32, 3x3x1>) ->  !migraphx.shaped<1x1x3xf32, 3x3x1> attributes {kernel, arch="gfx950"}{
  %result = migraphx.greater %arg0, %arg1: <1x1x3xf32, 3x3x1>, <1x1x3xf32, 3x3x1> -> <1x1x3xf32, 3x3x1>
  func.return %result : !migraphx.shaped<1x1x3xf32, 3x3x1>
}

// CHECK-LABEL: @func_equal(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[zero:.*]] = tensor.empty
// CHECK-DAG: %[[one:.*]] = linalg.generic {{.*}} ins(%[[expanded_0]], %[[expanded]] : tensor{{.*}}) outs(%[[zero]] : tensor{{.*}}) {
// CHECK-DAG:    ^bb0(%[[in:.*]]: {{.*}}, %[[in_1:.*]]: {{.*}}, %[[out:.*]]: {{.*}}):
// CHECK-DAG:        %[[two:.*]] = arith.cmpi eq, %[[in]], %[[in_1]]
// CHECK-DAG:        %[[three:.*]] = arith.extui %[[two]] : i1 to i32
// CHECK-DAG:        linalg.yield %[[three]]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[one]]
// CHECK-DAG: return %[[collapsed]]
func.func @func_equal(%arg0: !migraphx.shaped<1x1x3xi32, 3x3x1>, %arg1: !migraphx.shaped<1x1x3xi32, 3x3x1>) ->  !migraphx.shaped<1x1x3xi32, 3x3x1> attributes {kernel, arch="gfx950"}{
  %result = migraphx.equal %arg0, %arg1: <1x1x3xi32, 3x3x1>, <1x1x3xi32, 3x3x1> -> <1x1x3xi32, 3x3x1>
  func.return %result : !migraphx.shaped<1x1x3xi32, 3x3x1>
}

// CHECK-LABEL: @func_equal_float(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[zero:.*]] = tensor.empty
// CHECK-DAG: %[[one:.*]] = linalg.generic {{.*}} ins(%[[expanded_0]], %[[expanded]] : tensor{{.*}}) outs(%[[zero]] : tensor{{.*}}) {
// CHECK-DAG:    ^bb0(%[[in:.*]]: {{.*}}, %[[in_1:.*]]: {{.*}}, %[[out:.*]]: {{.*}}):
// CHECK-DAG:        %[[two:.*]] = arith.cmpf oeq, %[[in]], %[[in_1]]
// CHECK-DAG:        %[[three:.*]] = arith.uitofp %[[two]]
// CHECK-DAG:        linalg.yield %[[three]]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[one]]
// CHECK-DAG: return %[[collapsed]]
func.func @func_equal_float(%arg0: !migraphx.shaped<1x1x3xf32, 3x3x1>, %arg1: !migraphx.shaped<1x1x3xf32, 3x3x1>) ->  !migraphx.shaped<1x1x3xf32, 3x3x1> attributes {kernel, arch="gfx950"}{
  %result = migraphx.equal %arg0, %arg1: <1x1x3xf32, 3x3x1>, <1x1x3xf32, 3x3x1> -> <1x1x3xf32, 3x3x1>
  func.return %result : !migraphx.shaped<1x1x3xf32, 3x3x1>
}
