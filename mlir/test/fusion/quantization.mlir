// RUN: rocmlir-driver -host-pipeline partition,highlevel -arch gfx906 %s | FileCheck %s

// CHECK-LABEL: test_conv_with_cast
// CHECK: arith.sitofp {{.*}} : i32 to f32

func.func @test_conv_with_cast(
    %input: tensor<1x8x8x4xi8>,
    %filter: tensor<8x1x1x4xi8>,
    %scale: tensor<8xf32>,
    %bias: tensor<8xi32>) -> tensor<1x8x8x8xf32>
{
    %zero = arith.constant dense<0> : tensor<8xi8>
    %output = "tosa.conv2d"(%input, %filter, %zero) {quantization_info = #tosa.conv_quant<input_zp = 0, weight_zp = 0>, arch = "gfx906", dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x8x8x4xi8>, tensor<8x1x1x4xi8>, tensor<8xi8>) -> tensor<1x8x8x8xi32>
    %output_cast = "tosa.cast"(%output) : (tensor<1x8x8x8xi32>) -> tensor<1x8x8x8xf32>
    return %output_cast : tensor<1x8x8x8xf32>
}

// CHECK-LABEL: test_dequantization_migraphx
// CHECK: arith.sitofp {{.*}} : i32 to f32
func.func @test_dequantization_migraphx(
    %input: tensor<1x8x8x4xi8>,
    %filter: tensor<8x1x1x4xi8>,
    %scale: tensor<8xf32>,
    %bias: tensor<8xi32>) -> tensor<1x8x8x8xf32>
{
    %zero = arith.constant dense<0> : tensor<8xi8>
    %output = "tosa.conv2d"(%input, %filter, %zero) {quantization_info = #tosa.conv_quant<input_zp = 0, weight_zp = 0>, arch = "gfx906", dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x8x8x4xi8>, tensor<8x1x1x4xi8>, tensor<8xi8>) -> tensor<1x8x8x8xi32>

    %shifted = "tosa.sub"(%output, %bias) {} : (tensor<1x8x8x8xi32>, tensor<8xi32>) -> tensor<1x8x8x8xi32>
    %shifted_cast = "tosa.cast"(%shifted) : (tensor<1x8x8x8xi32>) -> tensor<1x8x8x8xf32>
    %scaled = "tosa.mul"(%shifted_cast, %scale) {shift = 0 : i32} : (tensor<1x8x8x8xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
    return %scaled : tensor<1x8x8x8xf32>
}

