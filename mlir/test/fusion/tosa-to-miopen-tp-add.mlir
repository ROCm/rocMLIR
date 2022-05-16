// RUN: mlir-miopen-driver -host-pipeline highlevel -kernel-pipeline rocdl %s -o -| FileCheck %s

// CHECK-LABEL: test_fusion

func @test_fusion(%arg0: tensor<256x28x28x128xf32>, %arg1: tensor<64x28x28x128xf32>, %arg2: tensor<256x64x28x28xf32>) -> tensor<256x28x28x64xf32> attributes {kernel} {
    %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi64>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %cst_0) {dilation = [1, 1], expected_filter_layout = "kyxc", expected_input_layout = "nhwc", expected_output_layout = "nhwk", pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<256x28x28x128xf32>, tensor<64x28x28x128xf32>, tensor<1xf32>) -> tensor<256x28x28x64xf32>
    %1 = "tosa.transpose"(%arg2, %cst) : (tensor<256x64x28x28xf32>, tensor<4xi64>) -> tensor<256x28x28x64xf32>
    %2 = "tosa.add"(%0, %1) : (tensor<256x28x28x64xf32>, tensor<256x28x28x64xf32>) -> tensor<256x28x28x64xf32>
    return %2 : tensor<256x28x28x64xf32>
}

// -----
