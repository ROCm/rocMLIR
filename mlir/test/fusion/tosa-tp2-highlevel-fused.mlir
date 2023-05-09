// RUN: rocmlir-driver --host-pipeline highlevel %s | FileCheck %s

// CHECK: rock.conv2d(%{{.*}}, %{{.*}}, %{{.*}}) features =  dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x", "g"], input_layout = ["ni", "hi", "wi", "ci", "gi"], output_layout = ["no", "ho", "wo", "ko", "go"], padding = [1 : i32, 1 : i32, 1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : memref<64x128x3x3x1xf32>, memref<256x28x28x128x1xf32>, memref<256x28x28x64x1xf32>

// CHECK-COUNT-1: linalg.generic
// CHECK-NOT: linalg.generic

module {
  func.func @test_fusion(%arg0: tensor<256x28x28x128xf32>, %arg1: tensor<64x128x3x3xf32>, %arg2: tensor<256x64x28x28xf32>) -> tensor<256x28x28x64xf32> attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx906"} {
    %cst_t = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi64>
    %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi64>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %a = "tosa.transpose"(%arg0, %cst_t) : (tensor<256x28x28x128xf32>, tensor<4xi64>) -> tensor<256x128x28x28xf32>
    %a2 = "tosa.transpose"(%a, %cst) : (tensor<256x128x28x28xf32>, tensor<4xi64>) -> tensor<256x28x28x128xf32>
    %b = "tosa.transpose"(%arg1, %cst) : (tensor<64x128x3x3xf32>, tensor<4xi64>) -> tensor<64x3x3x128xf32>
    %0 = "tosa.conv2d"(%a2, %b, %cst_0) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<256x28x28x128xf32>, tensor<64x3x3x128xf32>, tensor<1xf32>) -> tensor<256x28x28x64xf32>

    %1 = "tosa.transpose"(%arg2, %cst) : (tensor<256x64x28x28xf32>, tensor<4xi64>) -> tensor<256x28x28x64xf32>
    %2 = "tosa.add"(%0, %1) : (tensor<256x28x28x64xf32>, tensor<256x28x28x64xf32>) -> tensor<256x28x28x64xf32>

    return %2 : tensor<256x28x28x64xf32>
  }
}
