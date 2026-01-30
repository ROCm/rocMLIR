// RUN: rocmlir-gen -fut dot_add -arch %arch --clone-harness %s |\
// RUN:     rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx-linalg,highlevel-linalg -targets %arch |\
// RUN:     rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut dot_add_wrapper --verifier clone - |\
// RUN:     rocmlir-driver -host-pipeline mhal -kernel-pipeline full |\
// RUN:     xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void |\
// RUN:     FileCheck %s

// CHECK: [1 1 1]
func.func @dot_add(%arg0 : !migraphx.shaped<1x3x2xf32, 6x2x1>, %arg1: !migraphx.shaped<1x2x3xf32, 6x3x1>)
  -> !migraphx.shaped<1x3x3xf32, 9x3x1> {
    %0 = migraphx.dot %arg0, %arg1 : <1x3x2xf32, 6x2x1>, <1x2x3xf32, 6x3x1> -> <1x3x3xf32, 9x3x1>
      func.return %0 : !migraphx.shaped<1x3x3xf32, 9x3x1>
}
