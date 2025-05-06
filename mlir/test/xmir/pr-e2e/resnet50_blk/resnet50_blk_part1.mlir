// RUN: rocmlir-gen -fut forward__part_1 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -fut forward__part_1_wrapper -rand 1 -rand_type float --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// ALLOW_RETRIES: 2 
// CHECK: [1 1 1]

module {
  func.func private @forward__part_1(%arg0: tensor<1x32x32x64xf32> {mhal.read_access}, %arg1: tensor<64x3x3x64xf32> {mhal.read_access}) -> (tensor<1x32x32x64xf32> {mhal.write_access}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.conv2d %arg0, %arg1, %cst, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x64xf32>
    %1 = tosa.clamp %0 {max_val = 6.000000e+00 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>
    return %1 : tensor<1x32x32x64xf32>
  }
}
