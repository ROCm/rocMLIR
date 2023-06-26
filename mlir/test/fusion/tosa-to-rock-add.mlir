// RUN: rocmlir-driver -host-pipeline highlevel -kernel-pipeline=gpu,rocdl -arch %arch %s -o -| FileCheck %s

// CHECK-LABEL: test_fusion

func.func @test_fusion(%arg0: tensor<128x32x32x8xf32>, %arg1: tensor<128x3x3x8xf32>, %arg3: tensor<128x30x30x128xf32>) -> tensor<128x30x30x128xf32> attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1030"} {
  %zero = arith.constant dense<0.0> : tensor<128xf32>
  %0 = "tosa.conv2d"(%arg0, %arg1, %zero) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<128x32x32x8xf32>, tensor<128x3x3x8xf32>, tensor<128xf32>) -> tensor<128x30x30x128xf32>
  %1 = "tosa.add"(%0, %arg3) {} : (tensor<128x30x30x128xf32>, tensor<128x30x30x128xf32>) -> tensor<128x30x30x128xf32>

  return %1 : tensor<128x30x30x128xf32>
}

// -----
