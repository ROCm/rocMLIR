// XFAIL: *
// RUN: mlir-miopen-driver -host-pipeline partition %s | miopen-opt --miopen-async-launch - | FileCheck %s


module {
// CHECK:  func private @mobilenetv1_outlined_part_0(%arg0: tensor<1x112x112x32xf32>, %arg1: tensor<3x3x32x1xf32>) -> tensor<1x112x112x32xf32> attributes {kernel} {
// CHECK:  func private @mobilenetv1_outlined_part_1(%arg0: tensor<1x112x112x32xf32>, %arg1: tensor<64x1x1x32xf32>) -> tensor<1x112x112x64xf32> attributes {kernel} {
// CHECK:  func @mobilenetv1(%arg0: tensor<1x112x112x32xf32>, %arg1: tensor<32x3x3x3xf32>, %arg2: tensor<3x3x32x1xf32>, %arg3: tensor<64x1x1x32xf32>) -> tensor<1x112x112x64xf32> {
// CHECK:    %token, %results = async.launch @mobilenetv1_outlined_part_0 (%arg0, %arg2) : (tensor<1x112x112x32xf32>, tensor<3x3x32x1xf32>) -> tensor<1x112x112x32xf32>
// CHECK:    %token_0, %results_1 = async.launch @mobilenetv1_outlined_part_1 [%token] (%results, %arg3) : (tensor<1x112x112x32xf32>, tensor<64x1x1x32xf32>) -> tensor<1x112x112x64xf32>
// CHECK:    async.await %token_0 : !async.token
// CHECK:    return %results_1 : tensor<1x112x112x64xf32>

  // TOSA Model Func
  func @mobilenetv1(%input_image: tensor<1x112x112x32xf32>, %f0: tensor<32x3x3x3xf32>, %f1: tensor<3x3x32x1xf32>, %f2: tensor<64x1x1x32xf32>) -> tensor<1x112x112x64xf32> {

    %bias0 = arith.constant dense<0.0> : tensor<32xf32>
    %bias1 = arith.constant dense<0.0> : tensor<64xf32>
    
    %dwconv0 = "tosa.depthwise_conv2d"(%input_image, %f1, %bias0) {
      dilation = [1, 1],
      pad = [1, 1, 1, 1],
      stride = [1, 1]
    }
     : (tensor<1x112x112x32xf32>, tensor<3x3x32x1xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>

    // matmul
    %conv1 = "tosa.conv2d"(%dwconv0, %f2, %bias1) {
      dilation = [1, 1],
      pad = [0, 0, 0, 0],
      stride = [1, 1]
    }
     : (tensor<1x112x112x32xf32>, tensor<64x1x1x32xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>

    %relu1 = "tosa.reluN"(%conv1) {
      max_fp = 6.0 : f32,
      max_int = 6 : i64
    }
     : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>

    // ... skip the rest of MobileNetv1

    return %relu1 : tensor<1x112x112x64xf32>
  }
}

