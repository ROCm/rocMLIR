// RUN: rocmlir-driver --host-pipeline highlevel %s | FileCheck %s
// CHECK: rock.conv(%{{.*}}, %{{.*}}, %{{.*}}) features =  dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "hi", "wi", "gi", "ci"], output_layout = ["no", "ho", "wo", "go", "ko"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : memref<1x64x128x3x3xf32>, memref<256x28x28x1x128xf32>, memref<256x28x28x1x64xf32>

// CHECK-COUNT-1: linalg.generic
// CHECK-NOT: linalg.generic

module {
  func.func @test_fusion(%arg0: tensor<256x28x28x128xf32>, %arg1: tensor<64x128x3x3xf32>, %arg2: tensor<256x64x28x28xf32>) -> tensor<256x28x28x64xf32> attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx906"} {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %a = "tosa.transpose"(%arg0) {perms = array<i32: 0, 3, 1, 2>} : (tensor<256x28x28x128xf32>) -> tensor<256x128x28x28xf32>
    %a2 = "tosa.transpose"(%a) {perms = array<i32: 0, 2, 3, 1>} : (tensor<256x128x28x28xf32>) -> tensor<256x28x28x128xf32>
    %b = "tosa.transpose"(%arg1) {perms = array<i32: 0, 2, 3, 1>} : (tensor<64x128x3x3xf32>) -> tensor<64x3x3x128xf32>
    %input_zp = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = "tosa.conv2d"(%a2, %b, %cst_0, %input_zp, %weight_zp) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<256x28x28x128xf32>, tensor<64x3x3x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<256x28x28x64xf32>

    %1 = "tosa.transpose"(%arg2) {perms = array<i32: 0, 2, 3, 1>} : (tensor<256x64x28x28xf32>) -> tensor<256x28x28x64xf32>
    %2 = "tosa.add"(%0, %1) : (tensor<256x28x28x64xf32>, tensor<256x28x28x64xf32>) -> tensor<256x28x28x64xf32>

    return %2 : tensor<256x28x28x64xf32>
  }
}
