// RUN: miopen-opt --migraphx-transform --canonicalize --migraphx-to-tosa %s -verify-diagnostics -o -| FileCheck %s

module  {
  // CHECK-LABEL: func @matmul
  // CHECK: tosa.matmul
  func @matmul(%arg0: tensor<2x256x384xf32>, %arg1: tensor<2x384x768xf32>) -> tensor<2x256x768xf32> {
    %0 = "migraphx.dot"(%arg0, %arg1) : (tensor<2x256x384xf32>, tensor<2x384x768xf32>) -> tensor<2x256x768xf32>
     return %0 : tensor<2x256x768xf32>
  }

  // CHECK-LABEL: func @matmul_larger_batch
  // CHECK: tosa.matmul
  func @matmul_larger_batch(%arg0: tensor<2x16x256x384xf32>, %arg1: tensor<2x16x384x768xf32>) -> tensor<2x16x256x768xf32> {
    %0 = "migraphx.dot"(%arg0, %arg1) : (tensor<2x16x256x384xf32>, tensor<2x16x384x768xf32>) -> tensor<2x16x256x768xf32>
     return %0 : tensor<2x16x256x768xf32>
  }

  // CHECK-LABEL: func @func_power
  // CHECK: tosa.pow
  func @func_power(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "migraphx.pow"(%arg0, %arg1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
     return %0 : tensor<16xf32>
  }

  // CHECK-LABEL: func @func_recip
  // CHECK: tosa.recip
  func @func_recip(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "migraphx.recip"(%arg0) : (tensor<16xf32>) -> tensor<16xf32>
     return %0 : tensor<16xf32>
  }

  // CHECK-LABEL: func @func_sqrt
  // CHECK: tosa.rsqrt
  // CHECK-NOT: tosa.reciprocal
  func @func_sqrt(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "migraphx.sqrt"(%arg0) : (tensor<16xf32>) -> tensor<16xf32>
    %1 = "migraphx.recip"(%0) : (tensor<16xf32>) -> tensor<16xf32>
     return %1 : tensor<16xf32>
  }
}

// -----

