// RUN:  rocmlir-driver -host-pipeline highlevel,partition -targets %arch %s| rocmlir-gen -ph -rand 1 -rand_type float -fut test_fusion  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full |xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2

// CHECK: [1 1 1]
func.func @test_fusion(%arg0: tensor<1x8x8x4xf32>, %arg1: tensor<8x1x1x4xf32>, %arg3: tensor<f32>) -> tensor<1x8x8x8xf32> {
  %zero = arith.constant dense<0.0> : tensor<8xf32>
  %0 = "tosa.conv2d"(%arg0, %arg1, %zero) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x8x8x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
  %2 = "tosa.add"(%0, %arg3) {} : (tensor<1x8x8x8xf32>, tensor<f32>) -> tensor<1x8x8x8xf32>

  return %2 : tensor<1x8x8x8xf32>
}

// -----

