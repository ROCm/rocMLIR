// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @dot_3D(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %0 = linalg.batch_matmul ins(%[[expanded_0]], %[[expanded]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
func.func @dot_3D(%arg0 : !migraphx.shaped<1x3x2xf32, 6x2x1>, %arg1: !migraphx.shaped<1x2x3xf32, 6x3x1>) 
  -> !migraphx.shaped<1x3x3xf32, 9x3x1>{
  %0 = migraphx.dot %arg0, %arg1 : <1x3x2xf32, 6x2x1>, <1x2x3xf32, 6x3x1> -> <1x3x3xf32, 9x3x1>
  func.return %0 : !migraphx.shaped<1x3x3xf32, 9x3x1> 
}

// -----

// CHECK-LABEL: func.func @dot_2D(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[zero:.*]] = linalg.batch_matmul ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[zero]]
// CHECK-DAG:     %[[collapsed_3:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:    return %[[collapsed_3]]
func.func @dot_2D(%arg0 : !migraphx.shaped<3x2xf32, 2x1>, %arg1: !migraphx.shaped<2x3xf32, 3x1>) 
  -> !migraphx.shaped<3x3xf32, 3x1>{
  %0 = migraphx.dot %arg0, %arg1 : <3x2xf32, 2x1>, <2x3xf32, 3x1> -> <3x3xf32, 3x1>
  func.return %0 : !migraphx.shaped<3x3xf32, 3x1> 
}
 
// -----

// CHECK-LABEL: func.func @dot_4D(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:       %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:       %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:       %[[collapsed:.*]] = tensor.collapse_shape %[[expanded_0]]
// CHECK-DAG:       %[[collapsed_1:.*]] = tensor.collapse_shape %[[expanded]]
// CHECK-DAG:       %[[cst:.*]] = arith.constant dense
// CHECK-DAG:       %[[zero:.*]] = linalg.batch_matmul ins(%[[collapsed]], %[[collapsed_1]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-DAG:       %[[expanded_2:.*]] = tensor.expand_shape %[[zero]]
// CHECK-DAG:       %[[collapsed_3:.*]] = tensor.collapse_shape %[[expanded_2]]
// CHECK-DAG:       return %[[collapsed_3]]
func.func @dot_4D(%arg0 : !migraphx.shaped<1x1x3x2xf32, 6x6x2x1>, %arg1: !migraphx.shaped<1x1x2x3xf32, 6x6x3x1>) 
  -> !migraphx.shaped<1x1x3x3xf32, 9x9x3x1>{
  %0 = migraphx.dot %arg0, %arg1 : <1x1x3x2xf32, 6x6x2x1>, <1x1x2x3xf32, 6x6x3x1> -> <1x1x3x3xf32, 9x9x3x1>
  func.return %0 : !migraphx.shaped<1x1x3x3xf32, 9x9x3x1> 
}

// -----

// CHECK-LABEL: func.func @dot_broadcast(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:       %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:       %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:       %[[collapsed:.*]] = tensor.collapse_shape %[[expanded_0]]
// CHECK-DAG:       %[[collapsed_1:.*]] = tensor.collapse_shape %[[expanded]]
// CHECK-DAG:       %[[cst:.*]] = arith.constant dense
// CHECK-DAG:       %[[zero:.*]] = linalg.batch_matmul ins(%[[collapsed]], %[[collapsed_1]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-DAG:       %[[expanded_2:.*]] = tensor.expand_shape %[[zero]]
// CHECK-DAG:       %[[collapsed_3:.*]] = tensor.collapse_shape %[[expanded_2]]
// CHECK-DAG:       return %[[collapsed_3]]
func.func @dot_broadcast(%arg0: !migraphx.shaped<3x2x2x2xf32, 8x4x2x1>, %arg1: !migraphx.shaped<2x3x2x2xf32, 12x4x2x1>)
    -> !migraphx.shaped<3x2x2x2xf32, 8x4x2x1> attributes {kernel, arch="gfx950"} {
  %0 = migraphx.dot %arg0, %arg1 : <3x2x2x2xf32, 8x4x2x1>, <2x3x2x2xf32, 12x4x2x1> -> <3x2x2x2xf32, 8x4x2x1>
  func.return %0 : !migraphx.shaped<3x2x2x2xf32, 8x4x2x1>
}

// -----

// taken from migraphx-to-tosa.mlir
// checking for the perf_config attributes as well
// CHECK-LABEL: func.func @dot_f16(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:       %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:       %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:       %[[collapsed:.*]] = tensor.collapse_shape %[[expanded_0]]
// CHECK-DAG:       %[[collapsed_1:.*]] = tensor.collapse_shape %[[expanded]]
// CHECK-DAG:       %[[cst:.*]] = arith.constant dense
// CHECK-DAG:       %[[zero:.*]] = linalg.batch_matmul {perf_config = "v2:16,16,8,16,16,4,1,1,1"} ins(%[[collapsed]], %[[collapsed_1]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-DAG:       %[[expanded_2:.*]] = tensor.expand_shape %[[zero]]
// CHECK-DAG:       %[[collapsed_3:.*]] = tensor.collapse_shape %[[expanded_2]]
// CHECK-DAG:       return %[[collapsed_3]]
func.func @dot_f16(%arg0: !migraphx.shaped<8x64x64x320xf16, 1310720x20480x320x1>, %arg1: !migraphx.shaped<8x64x320x320xf16, 6553600x102400x320x1>) -> !migraphx.shaped<8x64x64x320xf16, 1310720x20480x320x1>{
  %4 = migraphx.dot %arg0, %arg1 {perf_config = "v2:16,16,8,16,16,4,1,1,1"} : <8x64x64x320xf16, 1310720x20480x320x1>, <8x64x320x320xf16, 6553600x102400x320x1> -> <8x64x64x320xf16, 1310720x20480x320x1>
  return %4 : !migraphx.shaped<8x64x64x320xf16, 1310720x20480x320x1>
}

// -----

func.func @dot_broadcast(%arg0: !migraphx.shaped<3x2x2x2xf32, 8x4x2x1>, %arg1: !migraphx.shaped<6x2x2xf32, 4x2x1>) -> !migraphx.shaped<3x2x2x2xf32, 8x4x2x1> attributes {kernel, arch="gfx950"} {
  // expected-error @+2 {{operands must have the same rank}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.dot' that was explicitly marked illegal}}
  %0 = migraphx.dot %arg0, %arg1 : <3x2x2x2xf32, 8x4x2x1>, <6x2x2xf32, 4x2x1> -> <3x2x2x2xf32, 8x4x2x1>
  func.return %0 : !migraphx.shaped<3x2x2x2xf32, 8x4x2x1>
}

// -----

func.func @dot_unranked_tensor(%arg0 : !migraphx.shaped<?x?x?xf32, ?x?x?>, %arg1: !migraphx.shaped<?x?x?xf32, ?x?x?>) -> !migraphx.shaped<?x?x?xf32, ?x?x?> {
    // expected-error @+2 {{only static shape is supported for now}}
    // expected-error @+1 {{failed to legalize operation 'migraphx.dot' that was explicitly marked illegal}}
    %0 = migraphx.dot %arg0, %arg1 : <?x?x?xf32, ?x?x?>, <?x?x?xf32, ?x?x?> -> <?x?x?xf32, ?x?x?>
    func.return %0 : !migraphx.shaped<?x?x?xf32, ?x?x?>
}
