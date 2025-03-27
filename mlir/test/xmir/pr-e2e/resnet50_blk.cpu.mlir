// RUN: rocmlir-driver -host-pipeline highlevel -targets %arch %s | rocmlir-gen -ph -print-results -fut resnet50 - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full -targets %arch | xmir-runner --target-type CPU --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2

module {
// CHECK: Unranked Memref base@ = 0x{{.*}} rank = 4 offset = 0 sizes = [1, 32, 32, 64] strides = [65536, 2048, 64, 1] data =


  func.func @resnet50(%arg0: tensor<1x32x32x64xf32>, %arg1: tensor<64x3x3x64xf32>, %arg2: tensor<64x3x3x64xf32>) -> tensor<1x32x32x64xf32> {

    %cst = arith.constant dense<0.0> : tensor<64xf32>
    %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %cst, %input_zp, %weight_zp) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x64xf32>
    %1 = "tosa.clamp"(%0) {min_val = 0.0 : f32, max_val = 6.0 : f32} : (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    %2 = "tosa.conv2d"(%1, %arg2, %cst, %input_zp, %weight_zp) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x64xf32>
    %3 = "tosa.clamp"(%2) {min_val = 0.0 : f32, max_val = 6.0 : f32} : (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>
    %4 = "tosa.add"(%arg0, %3) : (tensor<1x32x32x64xf32>, tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    return %4 : tensor<1x32x32x64xf32>
  }
}

