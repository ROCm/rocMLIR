// RUN: rocmlir-gen -fut test_fusion --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -print-inputs -print-results -fut test_fusion_wrapper -rand none - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext -entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2

// CHECK: Unranked Memref base
// CHECK: 5,     5,     5,     5,     5,     5,     5,     5
// CHECK-COUNT-504: 5
// Test first 8 and then remaining 504 out of total 512

func.func @test_fusion(%arg0: tensor<1x8x8x4xf32>, %arg1: tensor<8x1x1x4xf32>, %arg3: tensor<1x1x1x8xf32>) -> tensor<1x8x8x8xf32> attributes {kernel} {
  %zero = arith.constant dense<0.0> : tensor<8xf32>
  %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = "tosa.conv2d"(%arg0, %arg1, %zero, %input_zp, %weight_zp) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x8x8x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x8x8x8xf32>
  %2 = "tosa.add"(%0, %arg3) {} : (tensor<1x8x8x8xf32>, tensor<1x1x1x8xf32>) -> tensor<1x8x8x8xf32>

  return %2 : tensor<1x8x8x8xf32>
}

// -----

