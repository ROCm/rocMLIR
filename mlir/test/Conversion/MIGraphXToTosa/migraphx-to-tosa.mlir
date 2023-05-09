// RUN: rocmlir-opt --migraphx-to-tosa %s -verify-diagnostics -o -| FileCheck %s

module  {
  // CHECK-LABEL: func.func @ConvBias
  func.func @ConvBias(%arg0: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32> {
    %0 = "migraphx.constant"() {value = dense<1.000000e+00> : tensor<64x64x1x1xf32>} : () -> tensor<64x64x1x1xf32>
    %1 = "migraphx.convolution"(%arg0, %0) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %2 = "migraphx.constant"() {value = dense<2.000000e+00> : tensor<1x64x56x56xf32>} : () -> tensor<1x64x56x56xf32>
    %3 = "migraphx.add"(%1, %2) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
     return %3 : tensor<1x64x56x56xf32>
  }
  // CHECK-LABEL: func.func @ConvNoBias
  func.func @ConvNoBias(%arg0: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32> {
    %0 = "migraphx.constant"() {value = dense<3.000000e+00> : tensor<64x64x1x1xf32>} : () -> tensor<64x64x1x1xf32>
    %1 = "migraphx.convolution"(%arg0, %0) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<1x64x56x56xf32>
     return %1 : tensor<1x64x56x56xf32>
  }

}

// -----

