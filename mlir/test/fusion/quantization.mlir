// RUN: rocmlir-driver -host-pipeline highlevel -arch gfx906 %s | FileCheck %s

// CHECK-LABEL: test_conv_with_cast
// CHECK: arith.sitofp {{.*}} : i32 to f32

func.func @test_conv_with_cast(
    %input: tensor<1x8x8x4xi8>,
    %filter: tensor<8x1x1x4xi8>,
    %scale: tensor<8xf32>,
    %bias: tensor<8xi32>) -> tensor<1x8x8x8xf32> attributes {kernel}
{
    %zero = arith.constant dense<0> : tensor<8xi8>
    %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %weight_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %output = "tosa.conv2d"(%input, %filter, %zero, %input_zp, %weight_zp) {arch = "gfx906", acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x8x8x4xi8>, tensor<8x1x1x4xi8>, tensor<8xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x8xi32>
    %output_cast = "tosa.cast"(%output) : (tensor<1x8x8x8xi32>) -> tensor<1x8x8x8xf32>
    return %output_cast : tensor<1x8x8x8xf32>
}

// CHECK-LABEL: test_dequantization_migraphx
// CHECK: arith.sitofp {{.*}} : i32 to f32
func.func @test_dequantization_migraphx(
    %input: tensor<1x8x8x4xi8>,
    %filter: tensor<8x1x1x4xi8>,
    %scale: tensor<8xf32>,
    %bias: tensor<8xi32>) -> tensor<1x8x8x8xf32> attributes {kernel}
{
    %zero = arith.constant dense<0> : tensor<8xi8>
    %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %weight_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %output = "tosa.conv2d"(%input, %filter, %zero, %input_zp, %weight_zp) {arch = "gfx906", acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x8x8x4xi8>, tensor<8x1x1x4xi8>, tensor<8xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x8xi32>

    %const_shape = "tosa.const_shape"() { values = dense<[1, 1, 1, 8]> : tensor<4xindex> } : () -> !tosa.shape<4>
    %bias_reshaped = "tosa.reshape"(%bias, %const_shape) : (tensor<8xi32>, !tosa.shape<4>) -> tensor<1x1x1x8xi32>
    %shifted = "tosa.sub"(%output, %bias_reshaped) {} : (tensor<1x8x8x8xi32>, tensor<1x1x1x8xi32>) -> tensor<1x8x8x8xi32>
    %shifted_cast = "tosa.cast"(%shifted) : (tensor<1x8x8x8xi32>) -> tensor<1x8x8x8xf32>
    %scale_reshaped = "tosa.reshape"(%scale, %const_shape) : (tensor<8xf32>, !tosa.shape<4>) -> tensor<1x1x1x8xf32>
    %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %scaled = "tosa.mul"(%shifted_cast, %scale_reshaped, %shift) : (tensor<1x8x8x8xf32>, tensor<1x1x1x8xf32>, tensor<1xi8>) -> tensor<1x8x8x8xf32>
    return %scaled : tensor<1x8x8x8xf32>
}
