// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @dot_one(
// CHECK-SAME:   %[[arg0:.*]]: tensor<6xf32>,
// CHECK-SAME:   %[[arg1:.*]]: tensor<6xf32>) -> tensor<9xf32> {
// CHECK-NEXT:      %[[expanded:.*]] = tensor.expand_shape %[[arg1]] {{\[\[0, 1, 2\]\]}}
// CHECK-SAME:          output_shape {{\[1, 2, 3\]}} : tensor<6xf32> into tensor<1x2x3xf32>
// CHECK-NEXT:      %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]] {{\[\[0, 1, 2\]\]}}
// CHECK-SAME:          output_shape {{\[1, 3, 2\]}} : tensor<6xf32> into tensor<1x3x2xf32>
// CHECK-NEXT:      %[[cst:.*]] = arith.constant dense<0.000000e+00> : tensor<1x3x3xf32>
// CHECK-NEXT:      %[[zero:.*]] = linalg.batch_matmul ins(%[[expanded_0]], %[[expanded]]
// CHECK-SAME:                tensor<1x3x2xf32>, tensor<1x2x3xf32>) outs(%[[cst]] : tensor<1x3x3xf32>) -> tensor<1x3x3xf32>
// CHECK-NEXT:      %[[collapsed:.*]] = tensor.collapse_shape %[[zero]] 
// CHECK-NEXT:      return %[[collapsed]] : tensor<9xf32>
func.func @dot_one(%arg0 : !migraphx.shaped<1x3x2xf32, 6x2x1>, %arg1: !migraphx.shaped<1x2x3xf32, 6x3x1>) 
  -> !migraphx.shaped<1x3x3xf32, 9x3x1>{
  %0 = migraphx.dot %arg0, %arg1 : <1x3x2xf32, 6x2x1>, <1x2x3xf32, 6x3x1> -> <1x3x3xf32, 9x3x1>
  func.return %0 : !migraphx.shaped<1x3x3xf32, 9x3x1> 
}

// -----

// CHECK-LABEL: func.func @dot_two(
// CHECK-SAME:                     %[[arg0:.*]]: tensor<6xf32>,
// CHECK-SAME:                     %[[arg1:.*]]: tensor<6xf32>) -> tensor<9xf32> {
// CHECK-NEXT:      %[[expanded:.*]] = tensor.expand_shape %[[arg1]] 
// CHECK-SAME:            {{\[\[0, 1\]\]}} output_shape {{\[2, 3\]}} : tensor<6xf32> into tensor<2x3xf32>
// CHECK-NEXT:      %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]] 
// CHECK-SAME:          {{\[\[0, 1\]\]}} output_shape {{\[3, 2\]}} : tensor<6xf32> into tensor<3x2xf32>
// CHECK-NEXT:      %[[cst:.*]] = arith.constant dense<0.000000e+00> : tensor<3x3xf32>
// CHECK-NEXT:      %[[linout:.*]] = linalg.matmul ins(%[[expanded_0]], %[[expanded]] 
// CHECK-SAME:          : tensor<3x2xf32>, tensor<2x3xf32>) outs(%[[cst]] : tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK-NEXT:      %[[collapse:.*]] = tensor.collapse_shape %[[arg0:.*]] {{\[\[0, 1\]\]}} : tensor<3x3xf32> into tensor<9xf32>
// CHECK-NEXT:      return %[[collapse]] : tensor<9xf32>
func.func @dot_two(%arg0 : !migraphx.shaped<3x2xf32, 2x1>, %arg1: !migraphx.shaped<2x3xf32, 3x1>) 
  -> !migraphx.shaped<3x3xf32, 3x1>{
  %0 = migraphx.dot %arg0, %arg1 : <3x2xf32, 2x1>, <2x3xf32, 3x1> -> <3x3xf32, 3x1>
  func.return %0 : !migraphx.shaped<3x3xf32, 3x1> 
}
 
// -----

func.func @dot_three(%arg0 : !migraphx.shaped<1x1x3x2xf32, 6x6x2x1>, %arg1: !migraphx.shaped<1x1x2x3xf32, 6x6x3x1>) 
  -> !migraphx.shaped<1x1x3x3xf32, 9x9x3x1>{
  // expected-error @+2 {{only support 2D/3D for now}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.dot' that was explicitly marked illegal}}
  %0 = migraphx.dot %arg0, %arg1 : <1x1x3x2xf32, 6x6x2x1>, <1x1x2x3xf32, 6x6x3x1> -> <1x1x3x3xf32, 9x9x3x1>
  func.return %0 : !migraphx.shaped<1x1x3x3xf32, 9x9x3x1> 
}
