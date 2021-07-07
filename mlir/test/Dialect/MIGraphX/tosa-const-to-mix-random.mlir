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
}

