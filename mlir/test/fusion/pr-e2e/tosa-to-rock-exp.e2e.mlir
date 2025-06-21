// RUN: rocmlir-gen -fut test_fusion --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -fut test_fusion_wrapper -rand 1 -rand_type float --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext -entry-point-result=void | FileCheck %s

module {
// CHECK: RMS = {{.*}}e-09
// CHECK: [1 0 0]
  func.func @test_fusion(%arg0: tensor<128x32x32x8xf32>, %arg1: tensor<128x3x3x8xf32>) -> tensor<128x30x30x128xf32> {

    %zero = arith.constant dense<0.0> : tensor<128xf32>
    %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %zero, %input_zp, %weight_zp) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<128x32x32x8xf32>, tensor<128x3x3x8xf32>, tensor<128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<128x30x30x128xf32>
    %1 = "tosa.exp"(%0) : (tensor<128x30x30x128xf32>) -> tensor<128x30x30x128xf32>

    return %1 : tensor<128x30x30x128xf32>
  }

}
