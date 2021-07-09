module  {
  func @test_leaky_relu(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %1 = "tosa.const"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
    %2 = "tosa.reshape"(%1) {new_shape = [1, 1]} : (tensor<f32>) -> tensor<1x1xf32>
    %3 = "tosa.add"(%arg0, %2) {} : (tensor<4x4xf32>, tensor<1x1xf32>) -> tensor<4x4xf32>
    %4 = "tosa.mul"(%3, %2) {shift = 0 : i32} : (tensor<4x4xf32>, tensor<1x1xf32>) -> tensor<4x4xf32>
    %5 = "tosa.reshape"(%0) {new_shape = [1, 1]} : (tensor<f32>) -> tensor<1x1xf32>
    %6 = "tosa.greater_equal"(%arg0, %5) : (tensor<4x4xf32>, tensor<1x1xf32>) -> tensor<4x4xi1>
    %7 = "tosa.select"(%6, %arg0, %4) : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %7 : tensor<4x4xf32>
  }

  func @test_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<1x1x8x16xf32>) -> tensor<1x32x32x16xf32> {
    %0 = "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
    %1 = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi32>
    %2 = "tosa.transpose"(%arg1, %1) : (tensor<1x1x8x16xf32>, tensor<4xi32>) -> tensor<16x1x1x8xf32>
    %3 = "tosa.conv2d"(%arg0, %2, %0) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
    return %3 : tensor<1x32x32x16xf32>
  }
}

