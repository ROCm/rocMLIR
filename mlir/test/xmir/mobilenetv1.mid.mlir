// RUN: mlir-miopen-driver -host-pipeline partition,highlevel %s | FileCheck %s

module {

// CHECK: func private @mobilenetv1_outlined_part_0(%arg0: memref<1x224x224x3xf32>, %arg1: memref<32x3x3x3xf32>, %arg2: memref<1x112x112x32xf32>) attributes {kernel} {

// CHECK: func @mobilenetv1(%arg0: memref<1x224x224x3xf32>, %arg1: memref<32x3x3x3xf32>, %arg2: memref<3x3x32x1xf32>, %arg3: memref<64x1x1x32xf32>, %arg4: memref<3x3x64x1xf32>, %arg5: memref<128x1x1x64xf32>, %arg6: memref<1x56x56x128xf32>) {
// CHECK:   %token = async.launch @mobilenetv1_outlined_part_0 (%arg0, %arg1, %{{.*}}) : (memref<1x224x224x3xf32>, memref<32x3x3x3xf32>, memref<1x112x112x32xf32>)
// CHECK:   async.await %token : !async.token

// CHECK:   %token_0 = async.launch @mobilenetv1_outlined_part_1 (%{{.*}}, %arg3, %{{.*}}) : (memref<1x112x112x32xf32>, memref<64x1x1x32xf32>, memref<1x112x112x64xf32>)
// CHECK:   async.await %token_0 : !async.token
        
// CHECK:   %token_1 = async.launch @mobilenetv1_outlined_part_2 (%{{.*}}, %arg5, %arg6) : (memref<1x56x56x64xf32>, memref<128x1x1x64xf32>, memref<1x56x56x128xf32>)
// CHECK:   async.await %token_1 : !async.token
    
  func @mobilenetv1(%input_image: tensor<1x224x224x3xf32>, %f0: tensor<32x3x3x3xf32>, %f1: tensor<3x3x32x1xf32>, %f2: tensor<64x1x1x32xf32>, %f3: tensor<3x3x64x1xf32>, %f4: tensor<128x1x1x64xf32>) -> tensor<1x56x56x128xf32> {

    %bias0 = arith.constant dense<0.0> : tensor<32xf32>
    %bias1 = arith.constant dense<0.0> : tensor<64xf32>
    %bias2 = arith.constant dense<0.0> : tensor<128xf32>
    
    %conv0 = "tosa.conv2d"(%input_image, %f0, %bias0) {
      dilation = [1, 1],
      pad = [1, 1, 1, 1],
      stride = [2, 2]
    } : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>

    %relu0 = "tosa.reluN"(%conv0) {
      max_fp = 6.0 : f32,
      max_int = 6 : i64
    } : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>

    // depth-wise separable 1
    %dwconv1 = "tosa.depthwise_conv2d"(%relu0, %f1, %bias0) {
      dilation = [1, 1],
      pad = [1, 1, 1, 1],
      stride = [1, 1]
    } : (tensor<1x112x112x32xf32>, tensor<3x3x32x1xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>

    %relu1 = "tosa.reluN"(%dwconv1) {
      max_fp = 6.0 : f32,
      max_int = 6 : i64
    } : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>

    %conv1 = "tosa.conv2d"(%relu1, %f2, %bias1) {
      dilation = [1, 1],
      pad = [0, 0, 0, 0],
      stride = [1, 1]
    } : (tensor<1x112x112x32xf32>, tensor<64x1x1x32xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>

    %relu2 = "tosa.reluN"(%conv1) {
      max_fp = 6.0 : f32,
      max_int = 6 : i64
    } : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>

    // depth-wise separable 2
    %dwconv2 = "tosa.depthwise_conv2d"(%relu2, %f3, %bias1) {
      dilation = [1, 1],
      pad = [1, 1, 1, 1],
      stride = [2, 2]
    } : (tensor<1x112x112x64xf32>, tensor<3x3x64x1xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>

    %relu3 = "tosa.reluN"(%dwconv2) {
      max_fp = 6.0 : f32,
      max_int = 6 : i64
    } : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>

    // matmul
    %conv2 = "tosa.conv2d"(%relu3, %f4, %bias2) {
      dilation = [1, 1],
      pad = [0, 0, 0, 0],
      stride = [1, 1]
    } : (tensor<1x56x56x64xf32>, tensor<128x1x1x64xf32>, tensor<128xf32>) -> tensor<1x56x56x128xf32>

    %relu4 = "tosa.reluN"(%conv2) {
      max_fp = 6.0 : f32,
      max_int = 6 : i64
    } : (tensor<1x56x56x128xf32>) -> tensor<1x56x56x128xf32>

    // ...

    return %relu4 : tensor<1x56x56x128xf32>
  }
}

