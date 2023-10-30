// RUN: rocmlir-opt -tosa-to-rock %s | FileCheck %s
// CHECK-LABEL: @mlir_transpose_dot
func.func @mlir_transpose_dot(%arg0: tensor<1x1x5x4xf32>, %arg1: tensor<1x1x5x3xf32>) -> tensor<1x1x4x3xf32> attributes {arch = "gfx1100", kernel = "mixr", num_cu = 48 : i64} {  %cst = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi64>
  %0 = "tosa.transpose"(%arg0, %cst) : (tensor<1x1x5x4xf32>, tensor<4xi64>) -> tensor<1x1x4x5xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1], [2], [3]] : tensor<1x1x4x5xf32> into tensor<1x4x5xf32>
  %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1], [2], [3]] : tensor<1x1x5x3xf32> into tensor<1x5x3xf32>
  // CHECK: rock.gemm
  %1 = "tosa.matmul"(%collapsed, %collapsed_0) : (tensor<1x4x5xf32>, tensor<1x5x3xf32>) -> tensor<1x4x3xf32>
  %expanded = tensor.expand_shape %1 [[0, 1], [2], [3]] : tensor<1x4x3xf32> into tensor<1x1x4x3xf32>
  return %expanded : tensor<1x1x4x3xf32>
}
