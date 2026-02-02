// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --linalg-to-rock -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @matmul_3D
// CHECK-NEXT: %[[expanded:.*]] = tensor.expand_shape
// CHECK-NEXT: %[[expanded_0:.*]] = tensor.expand_shape
// CHECK-NEXT: %[[cst:.*]] = arith.constant
// CHECK-NEXT: %[[zero:.*]] = bufferization.alloc_tensor
// CHECK-NEXT: %[[one:.*]] = rock.gemm %[[zero]] = %[[expanded_0]] * %[[expanded]] storeMethod =  set
// CHECK-NEXT: %[[collapsed:.*]] = tensor.collapse_shape %[[one]]
// CHECK-NEXT: return %[[collapsed]]
func.func @matmul_3D(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<9xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [1, 2, 3] : tensor<6xf32> into tensor<1x2x3xf32>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [1, 3, 2] : tensor<6xf32> into tensor<1x3x2xf32>
  %cst = arith.constant dense<0.000000e+00> : tensor<1x3x3xf32>
  %0 = linalg.batch_matmul ins(%expanded_0, %expanded : tensor<1x3x2xf32>, tensor<1x2x3xf32>) outs(%cst : tensor<1x3x3xf32>) -> tensor<1x3x3xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<1x3x3xf32> into tensor<9xf32>
  return %collapsed : tensor<9xf32>
}

// CHECK-LABEL: func.func @matmul_2D
// CHECK-NEXT: %[[expanded:.*]] = tensor.expand_shape
// CHECK-NEXT: %[[expanded_0:.*]] = tensor.expand_shape
// CHECK-NEXT: %[[cst:.*]] = arith.constant
// CHECK-NEXT: %[[zero:.*]] = bufferization.alloc_tensor
// CHECK-NEXT: %[[one:.*]] = rock.gemm %[[zero]] = %[[expanded_0]] * %[[expanded]] storeMethod =  set
// CHECK-NEXT: %[[collapsed:.*]] = tensor.collapse_shape %[[one]]
// CHECK-NEXT: return %[[collapsed]]
func.func @matmul_2D(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<9xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1]] output_shape [3, 2] : tensor<6xf32> into tensor<3x2xf32>
  %cst = arith.constant dense<0.000000e+00> : tensor<3x3xf32>
  %0 = linalg.matmul ins(%expanded_0, %expanded : tensor<3x2xf32>, tensor<2x3xf32>) outs(%cst : tensor<3x3xf32>) -> tensor<3x3xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<3x3xf32> into tensor<9xf32>
  return %collapsed : tensor<9xf32>
}

