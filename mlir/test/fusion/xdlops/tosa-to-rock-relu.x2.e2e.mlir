// RUN: rocmlir-driver -host-pipeline highlevel --targets=%arch %s | rocmlir-gen -ph -print-results -rand 1 -rand_type float - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {
// CHECK: Unranked Memref base
  func.func @test_fusion(%arg0: tensor<128x32x32x8xf32>, %arg1: tensor<128x3x3x8xf32>) -> tensor<128x30x30x128xf32> attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx908"} {

    %zero = arith.constant dense<0.0> : tensor<128xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %zero) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, xdlopsV2 = true} : (tensor<128x32x32x8xf32>, tensor<128x3x3x8xf32>, tensor<128xf32>) -> tensor<128x30x30x128xf32>
    %1 = "tosa.clamp"(%0) {min_fp = 0.0 : f32, max_fp = 6.0 : f32, min_int = 0 : i64, max_int = 6 : i64} : (tensor<128x30x30x128xf32>) -> tensor<128x30x30x128xf32>
    return %1 : tensor<128x30x30x128xf32>
  }

}

