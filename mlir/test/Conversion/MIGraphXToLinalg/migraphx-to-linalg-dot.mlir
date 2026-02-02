// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @dot_3D
// CHECK-NEXT:  %[[expanded:.*]] = tensor.expand_shape
// CHECK-NEXT:  %[[expanded_0:.*]] = tensor.expand_shape
// CHECK-NEXT:  %[[cst:.*]] = arith.constant dense
// CHECK-NEXT:  %[[zero:.*]] = linalg.batch_matmul ins(%[[expanded_0]], %[[expanded]] {{.*}})
// CHECK-NEXT:  %[[collapsed:.*]] = tensor.collapse_shape %[[zero]]
// CHECK-NEXT:   return %[[collapsed]]
func.func @dot_3D(%arg0 : !migraphx.shaped<1x3x2xf32, 6x2x1>, %arg1: !migraphx.shaped<1x2x3xf32, 6x3x1>) 
  -> !migraphx.shaped<1x3x3xf32, 9x3x1>{
  %0 = migraphx.dot %arg0, %arg1 : <1x3x2xf32, 6x2x1>, <1x2x3xf32, 6x3x1> -> <1x3x3xf32, 9x3x1>
  func.return %0 : !migraphx.shaped<1x3x3xf32, 9x3x1> 
}

// -----

// CHECK-LABEL: func.func @dot_2D
// CHECK-NEXT:  %[[expanded:.*]] = tensor.expand_shape
// CHECK-NEXT:  %[[expanded_0:.*]] = tensor.expand_shape
// CHECK-NEXT:  %[[cst:.*]] = arith.constant dense
// CHECK-NEXT:  %[[zero:.*]] = linalg.matmul ins(%[[expanded_0]], %[[expanded]]{{.*}})
// CHECK-NEXT:  %[[collapsed:.*]] = tensor.collapse_shape %[[zero]]
// CHECK-NEXT:  return %[[collapsed]]
func.func @dot_2D(%arg0 : !migraphx.shaped<3x2xf32, 2x1>, %arg1: !migraphx.shaped<2x3xf32, 3x1>) 
  -> !migraphx.shaped<3x3xf32, 3x1>{
  %0 = migraphx.dot %arg0, %arg1 : <3x2xf32, 2x1>, <2x3xf32, 3x1> -> <3x3xf32, 3x1>
  func.return %0 : !migraphx.shaped<3x3xf32, 3x1> 
}
 
// -----

func.func @dot_4D(%arg0 : !migraphx.shaped<1x1x3x2xf32, 6x6x2x1>, %arg1: !migraphx.shaped<1x1x2x3xf32, 6x6x3x1>) 
  -> !migraphx.shaped<1x1x3x3xf32, 9x9x3x1>{
  // expected-error @+2 {{only support 2D/3D for now}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.dot' that was explicitly marked illegal}}
  %0 = migraphx.dot %arg0, %arg1 : <1x1x3x2xf32, 6x6x2x1>, <1x1x2x3xf32, 6x6x3x1> -> <1x1x3x3xf32, 9x9x3x1>
  func.return %0 : !migraphx.shaped<1x1x3x3xf32, 9x9x3x1> 
}

// -----

func.func @dot_unranked_tensor(%arg0 : !migraphx.shaped<?x?x?xf32, ?x?x?>, %arg1: !migraphx.shaped<?x?x?xf32, ?x?x?>) -> !migraphx.shaped<?x?x?xf32, ?x?x?> {
    // expected-error @+2 {{only static shape is supported for now}}
    // expected-error @+1 {{failed to legalize operation 'migraphx.dot' that was explicitly marked illegal}}
    %0 = migraphx.dot %arg0, %arg1 : <?x?x?xf32, ?x?x?>, <?x?x?xf32, ?x?x?> -> <?x?x?xf32, ?x?x?>
    func.return %0 : !migraphx.shaped<?x?x?xf32, ?x?x?>
}
