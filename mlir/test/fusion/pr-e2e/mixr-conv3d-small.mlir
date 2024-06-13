// RUN: rocmlir-gen -fut mlir_convolution_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_convolution_add_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
func.func private @mlir_convolution_add(%arg0: !migraphx.shaped<2x4x2x2x2xf32, 0x1x0x0x0>, %arg1: !migraphx.shaped<2x3x5x5x5xf32, 375x125x25x5x1>, %arg2: !migraphx.shaped<4x3x2x2x2xf32, 24x8x4x2x1>) -> !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1>  {
  %0 = migraphx.convolution %arg1, %arg2 {dilation = [2, 2, 2], group = 1 : i64, padding = [0, 0, 0, 0, 0, 0], padding_mode = 0 : i64, stride = [2, 2, 2]} : <2x3x5x5x5xf32, 375x125x25x5x1>, <4x3x2x2x2xf32, 24x8x4x2x1> -> <2x4x2x2x2xf32, 32x8x4x2x1>
  %1 = migraphx.add %0, %arg0 : <2x4x2x2x2xf32, 32x8x4x2x1>, <2x4x2x2x2xf32, 0x1x0x0x0> -> <2x4x2x2x2xf32, 32x8x4x2x1>
  return %1 : !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1>
}
