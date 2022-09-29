// RUN: rocmlir-opt --tosa-to-rock %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: test_fusion
func.func @test_fusion(%arg0: tensor<128x32x32x8xf32>, %arg1: tensor<128x3x3x8xf32>, %arg3: tensor<128x30x30x128xf32>) -> tensor<128x30x30x128xf32> attributes {kernel, arch = ""} {
  %zero = arith.constant dense<0.0> : tensor<128xf32>
  %0 = "tosa.conv2d"(%arg0, %arg1, %zero) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<128x32x32x8xf32>, tensor<128x3x3x8xf32>, tensor<128xf32>) -> tensor<128x30x30x128xf32>
  %1 = "tosa.abs"(%0) {} : (tensor<128x30x30x128xf32>) -> tensor<128x30x30x128xf32>
  %2 = "tosa.add"(%1, %arg3) {} : (tensor<128x30x30x128xf32>, tensor<128x30x30x128xf32>) -> tensor<128x30x30x128xf32>

  return %2 : tensor<128x30x30x128xf32>
}

// -----
