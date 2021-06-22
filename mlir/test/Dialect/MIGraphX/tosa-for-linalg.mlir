module  {
  func @test_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<1x1x8x16xf32>) -> tensor<1x32x32x16xf32> {
    %0 = "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
    %1 = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %2 = "tosa.transpose"(%arg1, %1) : (tensor<1x1x8x16xf32>, tensor<4xi32>) -> tensor<16x1x1x8xf32>
    %3 = "tosa.conv2d"(%arg0, %2, %0) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
    return %3 : tensor<1x32x32x16xf32>
  }
  func @test_depthwise_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<1x1x8x2xf32>) -> tensor<1x32x32x16xf32> {
    %0 = "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
    %1 = "tosa.depthwise_conv2d"(%arg0, %arg1, %0) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x32x32x8xf32>, tensor<1x1x8x2xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
    return %1 : tensor<1x32x32x16xf32>
  }
  func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_sub(%arg0: tensor<1x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.sub"(%arg0, %arg1) : (tensor<1x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_mul(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.mul"(%arg0, %arg1) {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_rcp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.reciprocal"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_relu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.reluN"(%arg0) {max_fp = 3.40282347E+38 : f32, max_int = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_relu6(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.reluN"(%arg0) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_leaky_relu(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %1 = "tosa.const"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
    %2 = "tosa.reshape"(%1) {new_shape = [1, 1]} : (tensor<f32>) -> tensor<1x1xf32>
    %3 = "tosa.mul"(%arg0, %2) {shift = 0 : i32} : (tensor<4x4xf32>, tensor<1x1xf32>) -> tensor<4x4xf32>
    %4 = "tosa.reshape"(%0) {new_shape = [1, 1]} : (tensor<f32>) -> tensor<1x1xf32>
    %5 = "tosa.greater_equal"(%arg0, %4) : (tensor<4x4xf32>, tensor<1x1xf32>) -> tensor<4x4xi1>
    %6 = "tosa.select"(%5, %arg0, %3) : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %6 : tensor<4x4xf32>
  }
  func @test_bitwise_and(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x3xi32> {
    %0 = "tosa.bitwise_and"(%arg0, %arg1) : (tensor<13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
    return %0 : tensor<13x21x3xi32>
  }
  func @test_bitwise_or(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>) -> tensor<13x21x3xi32> {
    %0 = "tosa.bitwise_or"(%arg0, %arg1) : (tensor<13x21x3xi32>, tensor<13x1x3xi32>) -> tensor<13x21x3xi32>
    return %0 : tensor<13x21x3xi32>
  }
  func @test_bitwise_not(%arg0: tensor<13x21x1xi32>) -> tensor<13x21x1xi32> {
    %0 = "tosa.bitwise_not"(%arg0) : (tensor<13x21x1xi32>) -> tensor<13x21x1xi32>
    return %0 : tensor<13x21x1xi32>
  }
  func @test_bitwise_xor(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
    %0 = "tosa.bitwise_xor"(%arg0, %arg1) : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
    return %0 : tensor<13x21x3xi32>
  }
  func @test_logical_and(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<13x21x3xi1> {
    %0 = "tosa.logical_and"(%arg0, %arg1) : (tensor<13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<13x21x3xi1>
    return %0 : tensor<13x21x3xi1>
  }
  func @test_logical_or(%arg0: tensor<13x1x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
    %0 = "tosa.logical_or"(%arg0, %arg1) : (tensor<13x1x3xi1>, tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
    return %0 : tensor<13x21x3xi1>
  }
  func @test_logical_not(%arg0: tensor<1x21x3xi1>) -> tensor<1x21x3xi1> {
    %0 = "tosa.logical_not"(%arg0) : (tensor<1x21x3xi1>) -> tensor<1x21x3xi1>
    return %0 : tensor<1x21x3xi1>
  }
  func @test_reduce_any(%arg0: tensor<13x21x3xi1>) -> tensor<21x3xi1> {
    %0 = "tosa.reduce_any"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xi1>) -> tensor<1x21x3xi1>
    %1 = "tosa.reshape"(%0) {new_shape = [21, 3]} : (tensor<1x21x3xi1>) -> tensor<21x3xi1>
    return %1 : tensor<21x3xi1>
  }
  func @test_reduce_all(%arg0: tensor<13x21x3xi1>) -> tensor<21x3xi1> {
    %0 = "tosa.reduce_all"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xi1>) -> tensor<1x21x3xi1>
    %1 = "tosa.reshape"(%0) {new_shape = [21, 3]} : (tensor<1x21x3xi1>) -> tensor<21x3xi1>
    return %1 : tensor<21x3xi1>
  }
  func @test_reduce_min(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
    %0 = "tosa.reduce_min"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
    %1 = "tosa.reshape"(%0) {new_shape = [21, 3]} : (tensor<1x21x3xf32>) -> tensor<21x3xf32>
    return %1 : tensor<21x3xf32>
  }
  func @test_reduce_max(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
    %0 = "tosa.reduce_max"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
    %1 = "tosa.reshape"(%0) {new_shape = [21, 3]} : (tensor<1x21x3xf32>) -> tensor<21x3xf32>
    return %1 : tensor<21x3xf32>
  }
  func @test_reduce_sum(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
    %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
    %1 = "tosa.reshape"(%0) {new_shape = [21, 3]} : (tensor<1x21x3xf32>) -> tensor<21x3xf32>
    return %1 : tensor<21x3xf32>
  }
  func @test_reduce_mean(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
    %0 = "tosa.const"() {value = dense<0.0769230798> : tensor<f32>} : () -> tensor<f32>
    %1 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
    %2 = "tosa.reshape"(%1) {new_shape = [21, 3]} : (tensor<1x21x3xf32>) -> tensor<21x3xf32>
    %3 = "tosa.reshape"(%0) {new_shape = [1, 1]} : (tensor<f32>) -> tensor<1x1xf32>
    %4 = "tosa.mul"(%2, %3) {shift = 0 : i32} : (tensor<21x3xf32>, tensor<1x1xf32>) -> tensor<21x3xf32>
    return %4 : tensor<21x3xf32>
  }
  func @test_reduce_product(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
    %0 = "tosa.reduce_prod"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
    %1 = "tosa.reshape"(%0) {new_shape = [21, 3]} : (tensor<1x21x3xf32>) -> tensor<21x3xf32>
    return %1 : tensor<21x3xf32>
  }
  func @test_min(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.minimum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_max(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.maximum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_pow(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.pow"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.abs"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.ceil"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_floor(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.floor"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_log(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.log"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_negate(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.negate"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.rsqrt"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_sigmoid(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.sigmoid"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_square(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.mul"(%arg0, %arg0) {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xi1> {
    %0 = "tosa.equal"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xi1>
    return %0 : tensor<13x21x3xi1>
  }
  func @test_greater_equal(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
    %0 = "tosa.greater_equal"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
    return %0 : tensor<13x21x3xi1>
  }
  func @test_greater(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
    %0 = "tosa.greater"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
    return %0 : tensor<13x21x3xi1>
  }
  func @test_less(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
    %0 = "tosa.greater_equal"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
    %1 = "tosa.logical_not"(%0) : (tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
    return %1 : tensor<13x21x3xi1>
  }
  func @test_less_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<13x21x3xi1> {
    %0 = "tosa.greater"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<13x21x3xi1>
    %1 = "tosa.logical_not"(%0) : (tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
    return %1 : tensor<13x21x3xi1>
  }
  func @test_argmax(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xi32> {
    %0 = "tosa.argmax"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<21x3xi32>
    return %0 : tensor<21x3xi32>
  }
  func @test_avg_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
    %0 = "tosa.avg_pool2d"(%arg0) {kernel = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
    return %0 : tensor<1x32x32x8xf32>
  }
  func @test_max_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
    %0 = "tosa.max_pool2d"(%arg0) {kernel = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
    return %0 : tensor<1x32x32x8xf32>
  }
  func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<1x819xf32> {
    %0 = "tosa.reshape"(%arg0) {new_shape = [1, 819]} : (tensor<13x21x3xf32>) -> tensor<1x819xf32>
    return %0 : tensor<1x819xf32>
  }
  func @test_transpose(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x21xf32> {
    %0 = "tosa.const"() {value = dense<[2, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %1 = "tosa.transpose"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21xf32>
    return %1 : tensor<3x13x21xf32>
  }
  func @test_select(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.const"() {value = dense<false> : tensor<1xi1>} : () -> tensor<1xi1>
    %1 = "tosa.reshape"(%0) {new_shape = [1, 1, 1]} : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %2 = "tosa.select"(%1, %arg0, %arg1) : (tensor<1x1x1xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %2 : tensor<13x21x3xf32>
  }
  func @test_addn(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    %1 = "tosa.add"(%arg2, %0) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    %2 = "tosa.add"(%arg3, %1) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %2 : tensor<13x21x3xf32>
  }
  func @test_pad(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.const"() {value = dense<0> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
    %1 = "tosa.pad"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<13x21x3xf32>
    return %1 : tensor<13x21x3xf32>
  }
  func @test_expand_dims(%arg0: tensor<13x21x3xf32>) -> tensor<1x13x21x3xf32> {
    %0 = "tosa.reshape"(%arg0) {new_shape = [1, 13, 21, 3]} : (tensor<13x21x3xf32>) -> tensor<1x13x21x3xf32>
    return %0 : tensor<1x13x21x3xf32>
  }
  func @test_shape() -> tensor<3xi32> {
    %0 = "tosa.const"() {value = dense<[13, 21, 3]> : tensor<3xi32>} : () -> tensor<3xi32>
    return %0 : tensor<3xi32>
  }
  func @test_rank() -> tensor<i32> {
    %0 = "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
    return %0 : tensor<i32>
  }
  func @test_elu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %1 = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %2 = "tosa.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    %3 = "tosa.reshape"(%0) {new_shape = [1, 1, 1]} : (tensor<f32>) -> tensor<1x1x1xf32>
    %4 = "tosa.sub"(%2, %3) : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
    %5 = "tosa.reshape"(%1) {new_shape = [1, 1, 1]} : (tensor<f32>) -> tensor<1x1x1xf32>
    %6 = "tosa.greater_equal"(%arg0, %5) : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xi1>
    %7 = "tosa.select"(%6, %arg0, %4) : (tensor<13x21x3xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %7 : tensor<13x21x3xf32>
  }
  func @test_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    %1 = "tosa.reduce_sum"(%0) {axis = 2 : i64} : (tensor<13x21x3xf32>) -> tensor<13x21x1xf32>
    %2 = "tosa.reciprocal"(%1) : (tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
    %3 = "tosa.mul"(%0, %2) {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
    return %3 : tensor<13x21x3xf32>
  }
  func @test_log_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    %1 = "tosa.reduce_sum"(%0) {axis = 2 : i64} : (tensor<13x21x3xf32>) -> tensor<13x21x1xf32>
    %2 = "tosa.reciprocal"(%1) : (tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
    %3 = "tosa.mul"(%0, %2) {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
    %4 = "tosa.log"(%3) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %4 : tensor<13x21x3xf32>
  }
  func @test_matmul(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>) -> tensor<14x28xf32> {
    %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<14x19xf32>, tensor<19x28xf32>) -> tensor<14x28xf32>
    return %0 : tensor<14x28xf32>
  }
  func @test_add_scalar(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %1 = "tosa.reshape"(%0) {new_shape = [1, 1, 1]} : (tensor<f32>) -> tensor<1x1x1xf32>
    %2 = "tosa.add"(%arg0, %1) : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
    return %2 : tensor<13x21x3xf32>
  }
  func @test_add_1d(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.reduce_sum"(%arg1) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
    %1 = "tosa.reduce_sum"(%0) {axis = 1 : i64} : (tensor<1x21x3xf32>) -> tensor<1x1x3xf32>
    %2 = "tosa.reshape"(%1) {new_shape = [3]} : (tensor<1x1x3xf32>) -> tensor<3xf32>
    %3 = "tosa.reshape"(%2) {new_shape = [1, 1, 3]} : (tensor<3xf32>) -> tensor<1x1x3xf32>
    %4 = "tosa.add"(%arg0, %3) : (tensor<13x21x3xf32>, tensor<1x1x3xf32>) -> tensor<13x21x3xf32>
    return %4 : tensor<13x21x3xf32>
  }
  func @test_tile(%arg0: tensor<13x21x3xf32>) -> tensor<39x21x6xf32> {
    %0 = "tosa.tile"(%arg0) {multiples = [3, 1, 2]} : (tensor<13x21x3xf32>) -> tensor<39x21x6xf32>
    return %0 : tensor<39x21x6xf32>
  }
  func @test_reverse(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.reverse"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_space_to_batch(%arg0: tensor<13x21x3xf32>) -> tensor<26x11x3xf32> {
    %0 = "tosa.const"() {value = dense<[[0, 0], [0, 1], [0, 0]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
    %1 = "tosa.const"() {value = dense<[2, 0, 1, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
    %2 = "tosa.pad"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<13x22x3xf32>
    %3 = "tosa.reshape"(%2) {new_shape = [13, 11, 2, 3]} : (tensor<13x22x3xf32>) -> tensor<13x11x2x3xf32>
    %4 = "tosa.transpose"(%3, %1) : (tensor<13x11x2x3xf32>, tensor<4xi32>) -> tensor<2x13x11x3xf32>
    %5 = "tosa.reshape"(%4) {new_shape = [26, 11, 3]} : (tensor<2x13x11x3xf32>) -> tensor<26x11x3xf32>
    return %5 : tensor<26x11x3xf32>
  }
  func @test_space_to_depth(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x16x16x32xf32> {
    %0 = "tosa.const"() {value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %1 = "tosa.reshape"(%arg0) {new_shape = [1, 16, 2, 16, 2, 8]} : (tensor<1x32x32x8xf32>) -> tensor<1x16x2x16x2x8xf32>
//    %2 = "tosa.transpose"(%1, %0) : (tensor<1x16x2x16x2x8xf32>, tensor<6xi32>) -> tensor<1x16x2x16x2x8xf32>
//    %3 = "tosa.reshape"(%2) {new_shape = [1, 16, 16, 32]} : (tensor<1x16x2x16x2x8xf32>) -> tensor<1x16x16x32xf32>
    %2 = "tosa.transpose"(%1, %0) : (tensor<1x16x2x16x2x8xf32>, tensor<6xi32>) -> tensor<1x16x16x2x2x8xf32>
    %3 = "tosa.reshape"(%2) {new_shape = [1, 16, 16, 32]} : (tensor<1x16x16x2x2x8xf32>) -> tensor<1x16x16x32xf32>
    return %3 : tensor<1x16x16x32xf32>
  }
  func @test_depth_to_space(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x64x64x2xf32> {
    %0 = "tosa.const"() {value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %1 = "tosa.reshape"(%arg0) {new_shape = [1, 32, 32, 2, 2, 2]} : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x2x2x2xf32>
    %2 = "tosa.transpose"(%1, %0) : (tensor<1x32x32x2x2x2xf32>, tensor<6xi32>) -> tensor<1x32x2x32x2x2xf32>
    %3 = "tosa.reshape"(%2) {new_shape = [1, 64, 64, 2]} : (tensor<1x32x2x32x2x2xf32>) -> tensor<1x64x64x2xf32>
//    %2 = "tosa.transpose"(%1, %0) : (tensor<1x32x32x2x2x2xf32>, tensor<6xi32>) -> tensor<1x32x32x2x2x2xf32>
//    %3 = "tosa.reshape"(%2) {new_shape = [1, 64, 64, 2]} : (tensor<1x32x32x2x2x2xf32>) -> tensor<1x64x64x2xf32>
    return %3 : tensor<1x64x64x2xf32>
  }
  func @test_left_shift(%arg0: tensor<4x4xi32>, %arg1: tensor<1x1xi32>) -> tensor<4x4xi32> {
    %0 = "tosa.logical_left_shift"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<1x1xi32>) -> tensor<4x4xi32>
    return %0 : tensor<4x4xi32>
  }
  func @test_right_shift(%arg0: tensor<4x4xi32>, %arg1: tensor<1x1xi32>) -> tensor<4x4xi32> {
    %0 = "tosa.arithmetic_right_shift"(%arg0, %arg1) {round = false} : (tensor<4x4xi32>, tensor<1x1xi32>) -> tensor<4x4xi32>
    return %0 : tensor<4x4xi32>
  }
  func @test_fakequant_with_min_max_args(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = "tosa.const"() {value = dense<-2.00003052> : tensor<f32>} : () -> tensor<f32>
    %1 = "tosa.const"() {value = dense<1.99996948> : tensor<f32>} : () -> tensor<f32>
    %2 = "tosa.const"() {value = dense<6.10360876E-5> : tensor<f32>} : () -> tensor<f32>
    %3 = "tosa.const"() {value = dense<16383.75> : tensor<f32>} : () -> tensor<f32>
    %4 = "tosa.const"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
    %5 = "tosa.reshape"(%1) {new_shape = [1, 1, 1]} : (tensor<f32>) -> tensor<1x1x1xf32>
    %6 = "tosa.minimum"(%arg0, %5) : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
    %7 = "tosa.reshape"(%0) {new_shape = [1, 1, 1]} : (tensor<f32>) -> tensor<1x1x1xf32>
    %8 = "tosa.maximum"(%6, %7) : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
    %9 = "tosa.reshape"(%0) {new_shape = [1, 1, 1]} : (tensor<f32>) -> tensor<1x1x1xf32>
    %10 = "tosa.sub"(%8, %9) : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
    %11 = "tosa.reshape"(%3) {new_shape = [1, 1, 1]} : (tensor<f32>) -> tensor<1x1x1xf32>
    %12 = "tosa.mul"(%10, %11) {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
    %13 = "tosa.reshape"(%4) {new_shape = [1, 1, 1]} : (tensor<f32>) -> tensor<1x1x1xf32>
    %14 = "tosa.add"(%12, %13) : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
    %15 = "tosa.floor"(%14) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    %16 = "tosa.reshape"(%2) {new_shape = [1, 1, 1]} : (tensor<f32>) -> tensor<1x1x1xf32>
    %17 = "tosa.mul"(%15, %16) {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
    %18 = "tosa.reshape"(%0) {new_shape = [1, 1, 1]} : (tensor<f32>) -> tensor<1x1x1xf32>
    %19 = "tosa.add"(%17, %18) : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
    return %19 : tensor<13x21x3xf32>
  }
  
}

