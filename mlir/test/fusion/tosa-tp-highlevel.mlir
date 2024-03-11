// RUN: rocmlir-driver --host-pipeline highlevel %s | FileCheck %s
// CHECK-COUNT-2: linalg.generic
// CHECK-NOT: linalg.generic
// This is only to detect any changes in tosa transpose optimization, nothing wrong it differs
// Just test needs to be amended once any change is detected.

module {
  func.func @layout_opt(%arg0: tensor<256x128x28x28xf32>, %arg1: tensor<64x128x3x3xf32>, %arg2: tensor<256x64x28x28xf32>
    , %arg3: tensor<64x64x3x3xf32>, %arg4: tensor<64xf32>) -> tensor<256x64x28x28xf32> {
    %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi64>
    %cst_1 = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi64>
    %0 = "tosa.transpose"(%arg0, %cst) : (tensor<256x128x28x28xf32>, tensor<4xi64>) -> tensor<256x28x28x128xf32>
    %1 = "tosa.transpose"(%arg1, %cst) : (tensor<64x128x3x3xf32>, tensor<4xi64>) -> tensor<64x3x3x128xf32>
    %2 = "tosa.conv2d"(%0, %1, %arg4) {dilation = array<i64: 1, 1>, expected_filter_layout = "kyxc", expected_input_layout = "nhwc", expected_output_layout = "nhwk", pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<256x28x28x128xf32>, tensor<64x3x3x128xf32>, tensor<64xf32>) -> tensor<256x28x28x64xf32>
    %3 = "tosa.transpose"(%2, %cst_1) : (tensor<256x28x28x64xf32>, tensor<4xi64>) -> tensor<256x64x28x28xf32>
    %4 = "tosa.add"(%3, %arg2) : (tensor<256x64x28x28xf32>, tensor<256x64x28x28xf32>) -> tensor<256x64x28x28xf32>
    %5 = "tosa.transpose"(%4, %cst) : (tensor<256x64x28x28xf32>, tensor<4xi64>) -> tensor<256x28x28x64xf32>
    %6 = "tosa.transpose"(%arg3, %cst) : (tensor<64x64x3x3xf32>, tensor<4xi64>) -> tensor<64x3x3x64xf32>
    %7 = "tosa.conv2d"(%5, %6, %arg4) {dilation = array<i64: 1, 1>, expected_filter_layout = "kyxc", expected_input_layout = "nhwc", expected_output_layout = "nhwk", pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<256x28x28x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<256x28x28x64xf32>
    %8 = "tosa.transpose"(%7, %cst_1) : (tensor<256x28x28x64xf32>, tensor<4xi64>) -> tensor<256x64x28x28xf32>
    return %8 : tensor<256x64x28x28xf32>
  }
}
