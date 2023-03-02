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

// CHECK-LABEL: test_quantization_ck
// CHECK: arith.sitofp {{.*}} : i32 to f32
// CHECK: arith.minf {{.*}} : f32
// CHECK: arith.maxf {{.*}} : f32
// CHECK: arith.fptosi {{.*}} : f32 to i8
// N, H, W, C = 1, 8, 8, 4
// K, Y, X, C = 8, 1, 1, 4
// N, H, W, K = 1, 8, 8, 8
func.func @test_quantization_ck(
    %input: tensor<1x8x8x4xi8>, 
    %filter: tensor<8x1x1x4xi8>, 
    %scale: tensor<8xf32>,
    %bias: tensor<8xi32>) -> tensor<1x8x8x8xi8> 
{
    %zero = arith.constant dense<0> : tensor<8xi8>  
    %output = "tosa.conv2d"(%input, %filter, %zero) {quantization_info = #tosa.conv_quant<input_zp = 0, weight_zp = 0>, arch = "gfx906", dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x8x8x4xi8>, tensor<8x1x1x4xi8>, tensor<8xi8>) -> tensor<1x8x8x8xi32>  
    %shifted = "tosa.add"(%output, %bias) {} : (tensor<1x8x8x8xi32>, tensor<8xi32>) -> tensor<1x8x8x8xi32>  
    %output_cast = "tosa.cast"(%shifted) : (tensor<1x8x8x8xi32>) -> tensor<1x8x8x8xf32>
    %scaled = "tosa.mul"(%output_cast, %scale) {shift = 0 : i32} : (tensor<1x8x8x8xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>  
    %scale_cast = "tosa.cast"(%scaled) : (tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xi8>
    return %scale_cast : tensor<1x8x8x8xi8>
}

// CHECK-LABEL: test_quantization_migraphx
// CHECK: arith.sitofp {{.*}} : i32 to f32
// CHECK: arith.minf {{.*}} : f32
// CHECK: arith.maxf {{.*}} : f32
// CHECK: arith.fptosi {{.*}} : f32 to i32
// CHECK: arith.trunci {{.*}} : i32 to i8
func.func @test_quantization_migraphx(
    %input: tensor<1x8x8x4xi8>, 
    %filter: tensor<8x1x1x4xi8>, 
    %scale: tensor<8xf32>,
    %bias: tensor<8xi32>) -> tensor<1x8x8x8xi8> 
{
    %zero = arith.constant dense<0> : tensor<8xi8>  
    %output = "tosa.conv2d"(%input, %filter, %zero) {quantization_info = #tosa.conv_quant<input_zp = 0, weight_zp = 0>, arch = "gfx906", dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x8x8x4xi8>, tensor<8x1x1x4xi8>, tensor<8xi8>) -> tensor<1x8x8x8xi32>  
    %output_cast = "tosa.cast"(%output) : (tensor<1x8x8x8xi32>) -> tensor<1x8x8x8xf32>
    %scaled = "tosa.mul"(%output_cast, %scale) {shift = 0 : i32} : (tensor<1x8x8x8xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>  
    %scaled_cast = "tosa.cast"(%scaled) : (tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xi32>
    %shifted = "tosa.add"(%scaled_cast, %bias) {} : (tensor<1x8x8x8xi32>, tensor<8xi32>) -> tensor<1x8x8x8xi32>  
    %scale_cast = "tosa.cast"(%shifted) : (tensor<1x8x8x8xi32>) -> tensor<1x8x8x8xi8>
    return %scale_cast : tensor<1x8x8x8xi8>
}

