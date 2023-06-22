// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets gfx908 %s | FileCheck %s

module {
// CHECK:  func.func private @mobilenetv1__part_1(%arg0: memref<1x112x112x64xf32> {func.read_access}, %arg1: memref<3x3x64x1xf32> {func.read_access}, %arg2: memref<1x56x56x64xf32> {func.write_access}) {
// CHECK:  func.func @mobilenetv1(%arg0: memref<1x224x224x3xf32>, %arg1: memref<32x3x3x3xf32>, %arg2: memref<3x3x32x1xf32>, %arg3: memref<64x1x1x32xf32>, %arg4: memref<3x3x64x1xf32>, %arg5: memref<128x1x1x64xf32>, %arg6: memref<1x56x56x128xf32>) {
// CHECK:  %[[TOKEN0:.*]] = mhal.launch @mobilenetv1__part_4 (%arg0, %arg1, %{{.*}}) : (memref<1x224x224x3xf32>, memref<32x3x3x3xf32>, memref<1x112x112x32xf32>)
// CHECK-DIS:   %[[TOKEN1:.*]] = mhal.launch @mobilenetv1__part_3 %[[TOKEN0:.*]] (%{{.*}}, %arg2, %{{.*}}) : (memref<1x112x112x32xf32>, memref<3x3x32x1xf32>, memref<1x112x112x32xf32>)
// CHECK:   mhal.await %token_{{.*}} : !mhal.token

  func.func @mobilenetv1(%input_image: tensor<1x224x224x3xf32>, %f0: tensor<32x3x3x3xf32>, %f1: tensor<3x3x32x1xf32>, %f2: tensor<64x1x1x32xf32>, %f3: tensor<3x3x64x1xf32>, %f4: tensor<128x1x1x64xf32>) -> tensor<1x56x56x128xf32> {

    %bias0 = arith.constant dense<0.0> : tensor<32xf32>
    %bias1 = arith.constant dense<0.0> : tensor<64xf32>
    %bias2 = arith.constant dense<0.0> : tensor<128xf32>

    %conv0 = "tosa.conv2d"(%input_image, %f0, %bias0) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>

    %relu0 = "tosa.clamp"(%conv0) {min_fp = 0.0 : f32, max_fp = 6.0 : f32, min_int = 0 : i64, max_int = 6 : i64} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>

    // depth-wise separable 1
    %dwconv1 = "tosa.depthwise_conv2d"(%relu0, %f1, %bias0) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x112x112x32xf32>, tensor<3x3x32x1xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
    %relu1 = "tosa.clamp"(%dwconv1) {min_fp = 0.0 : f32, max_fp = 6.0 : f32, min_int = 0 : i64, max_int = 6 : i64} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>

    %conv1 = "tosa.conv2d"(%relu1, %f2, %bias1) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x112x112x32xf32>, tensor<64x1x1x32xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
    %relu2 = "tosa.clamp"(%conv1) {min_fp = 0.0 : f32, max_fp = 6.0 : f32, min_int = 0 : i64, max_int = 6 : i64} : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>

    // depth-wise separable 2
    %dwconv2 = "tosa.depthwise_conv2d"(%relu2, %f3, %bias1) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x112x112x64xf32>, tensor<3x3x64x1xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %relu3 = "tosa.clamp"(%dwconv2) {min_fp = 0.0 : f32, max_fp = 6.0 : f32, min_int = 0 : i64, max_int = 6 : i64} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>

    // matmul
    %conv2 = "tosa.conv2d"(%relu3, %f4, %bias2) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x56x56x64xf32>, tensor<128x1x1x64xf32>, tensor<128xf32>) -> tensor<1x56x56x128xf32>
    %relu4 = "tosa.clamp"(%conv2) {min_fp = 0.0 : f32, max_fp = 6.0 : f32, min_int = 0 : i64, max_int = 6 : i64} : (tensor<1x56x56x128xf32>) -> tensor<1x56x56x128xf32>

    // ...

    return %relu4 : tensor<1x56x56x128xf32>
  }
}
