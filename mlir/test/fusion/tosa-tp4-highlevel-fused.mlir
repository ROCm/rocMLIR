// RUN: rocmlir-driver --host-pipeline highlevel %s | FileCheck %s

module {
  // CHECK-LABEL: @test_conv_tp
  func.func @test_conv_tp(%arg0: tensor<256x28x28x128xf32>, %arg1: tensor<64x128x3x3xf32>, %arg2: tensor<256x64x28x28xf32>) -> tensor<256x28x28x64xf32> attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx906"} {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %a = "tosa.transpose"(%arg0) {perms = array<i32: 0, 3, 1, 2>} : (tensor<256x28x28x128xf32>) -> tensor<256x128x28x28xf32>
    %a2 = "tosa.transpose"(%a) {perms = array<i32: 0, 2, 3, 1>} : (tensor<256x128x28x28xf32>) -> tensor<256x28x28x128xf32>
    %b = "tosa.transpose"(%arg1) {perms = array<i32: 0, 2, 3, 1>} : (tensor<64x128x3x3xf32>) -> tensor<64x3x3x128xf32>
    %input_zp = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: rock.conv(%{{.*}}, %{{.*}}, %{{.*}}) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "hi", "wi", "gi", "ci"], output_layout = ["no", "ho", "wo", "go", "ko"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : memref<1x64x128x3x3xf32>, memref<256x28x28x1x128xf32>, memref<256x28x28x1x64xf32>
    // CHECK-COUNT-1: linalg.generic
    // CHECK-NOT: linalg.generic
    %c0 = "tosa.conv2d"(%a2, %b, %cst_0, %input_zp, %weight_zp) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<256x28x28x128xf32>, tensor<64x3x3x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<256x28x28x64xf32>

    %c1 = "tosa.transpose"(%c0) {perms = array<i32: 0, 3, 1, 2>} : (tensor<256x28x28x64xf32>) -> tensor<256x64x28x28xf32>
    %c2 = "tosa.transpose"(%c1) {perms = array<i32: 0, 2, 3, 1>} : (tensor<256x64x28x28xf32>) -> tensor<256x28x28x64xf32>
    %1 = "tosa.transpose"(%arg2) {perms = array<i32: 0, 2, 3, 1>} : (tensor<256x64x28x28xf32>) -> tensor<256x28x28x64xf32>
    %2 = "tosa.add"(%c2, %1) : (tensor<256x28x28x64xf32>, tensor<256x28x28x64xf32>) -> tensor<256x28x28x64xf32>

    return %2 : tensor<256x28x28x64xf32>
  }

  // CHECK-LABEL: @test_conv_tp_reshape1
  func.func @test_conv_tp_reshape1(%arg0: tensor<1x256x28x28x128xf32>, %arg1: tensor<1x64x128x3x3xf32>, %arg2: tensor<1x256x64x28x28xf32>) -> tensor<1x256x28x28x64xf32> attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx906"} {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %a = "tosa.transpose"(%arg0) {perms = array<i32: 0, 1, 4, 2, 3>} : (tensor<1x256x28x28x128xf32>) -> tensor<1x256x128x28x28xf32>
    %a2 = "tosa.transpose"(%a) {perms = array<i32: 0, 1, 3, 4, 2>} : (tensor<1x256x128x28x28xf32>) -> tensor<1x256x28x28x128xf32>
    %b = "tosa.transpose"(%arg1) {perms = array<i32: 0, 1, 3, 4, 2>} : (tensor<1x64x128x3x3xf32>) -> tensor<1x64x3x3x128xf32>

    // CHECK: rock.conv(%{{.*}}, %{{.*}}, %{{.*}}) features =  dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "hi", "wi", "gi", "ci"], output_layout = ["no", "ho", "wo", "go", "ko"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : memref<1x64x128x3x3xf32>, memref<256x28x28x1x128xf32>, memref<256x28x28x1x64xf32>
    // CHECK-COUNT-1: linalg.generic
    // CHECK-NOT: linalg.generic
    %const_shape = "tosa.const_shape"() { value = dense<[256, 28, 28, 128]> : tensor<4xindex> } : () -> !tosa.shape<4>
    %a2_rshp = "tosa.reshape"(%a2, %const_shape) : (tensor<1x256x28x28x128xf32>, !tosa.shape<4>) -> tensor<256x28x28x128xf32>
    %const_shape2 = "tosa.const_shape"() { value = dense<[64, 3, 3, 128]> : tensor<4xindex> } : () -> !tosa.shape<4>
    %b_rshp = "tosa.reshape"(%b, %const_shape2) : (tensor<1x64x3x3x128xf32>, !tosa.shape<4>) -> tensor<64x3x3x128xf32>
    %input_zp = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %c0 = "tosa.conv2d"(%a2_rshp, %b_rshp, %cst_0, %input_zp, %weight_zp) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<256x28x28x128xf32>, tensor<64x3x3x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<256x28x28x64xf32>
    %const_shape3 = "tosa.const_shape"() { value = dense<[1, 256, 28, 28, 64]> : tensor<5xindex> } : () -> !tosa.shape<5>
    %c0_rshp = "tosa.reshape"(%c0, %const_shape3) : (tensor<256x28x28x64xf32>, !tosa.shape<5>) -> tensor<1x256x28x28x64xf32>

    %c1 = "tosa.transpose"(%c0_rshp) {perms = array<i32: 0, 1, 4, 2, 3>} : (tensor<1x256x28x28x64xf32>) -> tensor<1x256x64x28x28xf32>
    %c2 = "tosa.transpose"(%c1) {perms = array<i32: 0, 1, 3, 4, 2>} : (tensor<1x256x64x28x28xf32>) -> tensor<1x256x28x28x64xf32>
    %1 = "tosa.transpose"(%arg2) {perms = array<i32: 0, 1, 3, 4, 2>} : (tensor<1x256x64x28x28xf32>) -> tensor<1x256x28x28x64xf32>
    %2 = "tosa.add"(%c2, %1) : (tensor<1x256x28x28x64xf32>, tensor<1x256x28x28x64xf32>) -> tensor<1x256x28x28x64xf32>

    return %2 : tensor<1x256x28x28x64xf32>
  }

  // CHECK-LABEL: @test_conv_tp_reshape2
  func.func @test_conv_tp_reshape2(%arg0: tensor<256x28x28x128x1xf32>, %arg1: tensor<64x128x3x3x1xf32>, %arg2: tensor<256x64x28x28x1xf32>) -> tensor<256x28x28x64x1xf32> attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx906"} {
    %cst_t = arith.constant dense<[0, 3, 1, 2, 4]> : tensor<5xi32>
    %cst = arith.constant dense<[0, 2, 3, 1, 4]> : tensor<5xi32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %a = "tosa.transpose"(%arg0) {perms = array<i32: 0, 3, 1, 2, 4>} : (tensor<256x28x28x128x1xf32>) -> tensor<256x128x28x28x1xf32>
    %a2 = "tosa.transpose"(%a) {perms = array<i32: 0, 2, 3, 1, 4>} : (tensor<256x128x28x28x1xf32>) -> tensor<256x28x28x128x1xf32>
    %b = "tosa.transpose"(%arg1) {perms = array<i32: 0, 2, 3, 1, 4>} : (tensor<64x128x3x3x1xf32>) -> tensor<64x3x3x128x1xf32>

    // CHECK: rock.conv(%{{.*}}, %{{.*}}, %{{.*}}) features =  dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "hi", "wi", "gi", "ci"], output_layout = ["no", "ho", "wo", "go", "ko"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : memref<1x64x128x3x3xf32>, memref<256x28x28x1x128xf32>, memref<256x28x28x1x64xf32>
    // CHECK-COUNT-1: linalg.generic
    // CHECK-NOT: linalg.generic
    %const_shape = "tosa.const_shape"() { value = dense<[256, 28, 28, 128]> : tensor<4xindex> } : () -> !tosa.shape<4>
    %a2_rshp = "tosa.reshape"(%a2, %const_shape) : (tensor<256x28x28x128x1xf32>, !tosa.shape<4>) -> tensor<256x28x28x128xf32>
    %const_shape2 = "tosa.const_shape"() { value = dense<[64, 3, 3, 128]> : tensor<4xindex> } : () -> !tosa.shape<4>
    %b_rshp = "tosa.reshape"(%b, %const_shape2) : (tensor<64x3x3x128x1xf32>, !tosa.shape<4>) -> tensor<64x3x3x128xf32>
    %input_zp = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %c0 = "tosa.conv2d"(%a2_rshp, %b_rshp, %cst_0, %input_zp, %weight_zp) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<256x28x28x128xf32>, tensor<64x3x3x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<256x28x28x64xf32>
    %const_shape3 = "tosa.const_shape"() { value = dense<[256, 28, 28, 64, 1]> : tensor<5xindex> } : () -> !tosa.shape<5>
    %c0_rshp = "tosa.reshape"(%c0, %const_shape3) : (tensor<256x28x28x64xf32>, !tosa.shape<5>) -> tensor<256x28x28x64x1xf32>

    %c1 = "tosa.transpose"(%c0_rshp) {perms = array<i32: 0, 3, 1, 2, 4>} : (tensor<256x28x28x64x1xf32>) -> tensor<256x64x28x28x1xf32>
    %c2 = "tosa.transpose"(%c1) {perms = array<i32: 0, 2, 3, 1, 4>} : (tensor<256x64x28x28x1xf32>) -> tensor<256x28x28x64x1xf32>
    %1 = "tosa.transpose"(%arg2) {perms = array<i32: 0, 2, 3, 1, 4>} : (tensor<256x64x28x28x1xf32>) -> tensor<256x28x28x64x1xf32>
    %2 = "tosa.add"(%c2, %1) : (tensor<256x28x28x64x1xf32>, tensor<256x28x28x64x1xf32>) -> tensor<256x28x28x64x1xf32>

    return %2 : tensor<256x28x28x64x1xf32>
  }
}
