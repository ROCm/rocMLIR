// RUN: rocmlir-gen -fut conv_2d_group -arch %arch --clone-harness %s | rocmlir-driver --host-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type=float -fut conv_2d_group_wrapper --verifier clone -  | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=BOTH
// RUN: rocmlir-gen -fut conv_2d_group -arch %arch --clone-harness %s | rocmlir-driver --host-pipeline=migraphx,highlevel --kernel-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type=float -fut conv_2d_group_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=BOTH

// Here we are checking to see if conv_2d with non standard stride, dilation, and a group parameter matches the existing tosa pipeline
// Note - this array is quite large, so we are only checking a small subset

// BOTH: [5.83007, 7.83374, 8.46274, 9.03237, 6.51391, 7.75809, 9.73003, 8.48013, 8.15419, 9.9975, 7.50244, 7.11982, 6.58057
func.func @conv_2d_group(%in: !migraphx.shaped<2x4x123x124xf32, 61008x15252x124x1>, %fil: !migraphx.shaped<8x2x4x5xf32, 40x20x5x1>) -> !migraphx.shaped<2x8x27x19xf32, 4104x513x19x1> {
  %out = migraphx.convolution %in, %fil {dilation = [2, 3], group = 2 : i64, padding = [2, 2, 2, 2], padding_mode = 0 : i64, stride = [4, 5]} : 
    <2x4x123x124xf32, 61008x15252x124x1>, <8x2x4x5xf32, 40x20x5x1> -> <2x8x27x19xf32, 4104x513x19x1>
  func.return %out : !migraphx.shaped<2x8x27x19xf32, 4104x513x19x1>
}
