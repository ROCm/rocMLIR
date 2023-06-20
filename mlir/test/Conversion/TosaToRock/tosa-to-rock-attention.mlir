// RUN: rocmlir-opt --tosa-to-rock %s -verify-diagnostics -o -| FileCheck %s

// CHECK: rock.attention
func.func @self_attention(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x384x64xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x384xf32>) -> tensor<1x384x64xf32> attributes {kernel, arch = ""} {
  %cst = arith.constant dense<[0, 2, 1]> : tensor<3xi64>
  %0 = "tosa.transpose"(%arg1, %cst) : (tensor<1x384x64xf32>, tensor<3xi64>) -> tensor<1x64x384xf32>
  %1 = "tosa.matmul"(%arg0, %0) : (tensor<1x384x64xf32>, tensor<1x64x384xf32>) -> tensor<1x384x384xf32>
  %2 = "tosa.mul"(%1, %arg3) {shift = 0 : i32} : (tensor<1x384x384xf32>, tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %3 = "tosa.reduce_max"(%2) {axis = 1 : i64} : (tensor<1x384x384xf32>) -> tensor<1x1x384xf32>
  %4 = "tosa.sub"(%2, %3) : (tensor<1x384x384xf32>, tensor<1x1x384xf32>) -> tensor<1x384x384xf32>
  %5 = "tosa.exp"(%4) : (tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %6 = "tosa.reduce_sum"(%5) {axis = 1 : i64} : (tensor<1x384x384xf32>) -> tensor<1x1x384xf32>
  %7 = "tosa.reciprocal"(%6) : (tensor<1x1x384xf32>) -> tensor<1x1x384xf32>
  %8 = "tosa.mul"(%5, %7) {shift = 0 : i32} : (tensor<1x384x384xf32>, tensor<1x1x384xf32>) -> tensor<1x384x384xf32>
  %9 = "tosa.matmul"(%8, %arg2) : (tensor<1x384x384xf32>, tensor<1x384x64xf32>) -> tensor<1x384x64xf32>
  return %9 : tensor<1x384x64xf32>
}
