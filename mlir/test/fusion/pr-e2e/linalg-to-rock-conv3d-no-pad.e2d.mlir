// RUN: rocmlir-gen -fut conv3d_add -arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx-linalg,highlevel -kernel-pipeline=migraphx-linalg,highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut conv3d_add_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]
func.func @conv3d_add(%arg1: !migraphx.shaped<2x3x5x5x5xf32, 375x125x25x5x1>, %arg2: !migraphx.shaped<9x1x2x2x2xf32, 8x8x4x2x1>) -> !migraphx.shaped<2x9x2x2x2xf32, 72x8x4x2x1> {
  %0 = migraphx.convolution %arg1, %arg2 {dilation = [2, 2, 2], group = 3 : i64, padding = [0, 0, 0, 0, 0, 0], padding_mode = 0 : i64, stride = [2, 2, 2]} : <2x3x5x5x5xf32, 375x125x25x5x1>, <9x1x2x2x2xf32, 8x8x4x2x1> -> <2x9x2x2x2xf32, 72x8x4x2x1>
  return %0: !migraphx.shaped<2x9x2x2x2xf32, 72x8x4x2x1>
}
