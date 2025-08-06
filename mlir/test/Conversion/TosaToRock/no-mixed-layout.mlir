// RUN: rocmlir-driver -host-pipeline highlevel -targets amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack- %s | FileCheck %s

// CHECK: rock.conv({{.*}}) {{.*}}, filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"]{{.*}}}

module {
  func.func @test(%arg0: tensor<1x512x1x1xf32>, %arg1: tensor<1x384x28x28xf32>, %arg2: tensor<512x384x1x1xf32>) -> tensor<1x512x28x28xf32> attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %0 = "tosa.transpose"(%arg1) {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x384x28x28xf32>) -> tensor<1x28x28x384xf32>
    %1 = "tosa.transpose"(%arg2) {perms = array<i32: 0, 2, 3, 1>} : (tensor<512x384x1x1xf32>) -> tensor<512x1x1x384xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %2 = "tosa.conv2d"(%0, %1, %cst_0, %input_zp, %weight_zp) <{acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x28x28x384xf32>, tensor<512x1x1x384xf32>, tensor<512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x28x28x512xf32>
    %3 = "tosa.transpose"(%2) {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x28x28x512xf32>) -> tensor<1x512x28x28xf32>
    %4 = "tosa.add"(%3, %arg0) : (tensor<1x512x28x28xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %5 = "tosa.clamp"(%4) <{max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32}> : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    return %5 : tensor<1x512x28x28xf32>
  }
}
