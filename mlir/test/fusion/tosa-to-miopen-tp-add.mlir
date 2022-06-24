// RUN: mlir-miopen-driver --host-pipeline highlevel %s | miopen-opt --miopen-affix-params --miopen-lowering --miopen-lowering-step2 --canonicalize --miopen-linalg-align | FileCheck %s
// CHECK: miopen.transform %arg2
// CHECK-SAME: memref<256x64x28x28xf32> to memref<256x28x28x64x1xf32
// CHECK: linalg.generic
// CHECK-SAME: memref<1x8x8xf32, 5>
// to test transpose is converted as transform and fused.

func @test_fusion(%arg0: tensor<256x28x28x128xf32>, %arg1: tensor<64x3x3x128xf32>, %arg2: tensor<256x64x28x28xf32>) -> tensor<256x28x28x64xf32> attributes {kernel} {
    %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi64>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %cst_0) {dilation = [1, 1], expected_filter_layout = "kyxc", expected_input_layout = "nhwc", expected_output_layout = "nhwk", pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<256x28x28x128xf32>, tensor<64x3x3x128xf32>, tensor<1xf32>) -> tensor<256x28x28x64xf32>
    %1 = "tosa.transpose"(%arg2, %cst) : (tensor<256x64x28x28xf32>, tensor<4xi64>) -> tensor<256x28x28x64xf32>
    %2 = "tosa.add"(%0, %1) : (tensor<256x28x28x64xf32>, tensor<256x28x28x64xf32>) -> tensor<256x28x28x64xf32>
    return %2 : tensor<256x28x28x64xf32>
}

