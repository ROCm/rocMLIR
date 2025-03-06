// RUN: rocmlir-gen -fut dot_splitk_linear --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand none -fut dot_splitk_linear_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut dot_splitk_linear --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut dot_splitk_linear_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
module {
  // CHECK: [1 1 1]
  // CHECK: [1 1 1]
  // CHECK:  [21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21]
  // CHECK:  [63,  63,  63,  63,  63]

  // CLONE: [1 1 1]
  // CLONE: [1 1 1]

  func.func @dot_splitk_linear(%arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>, %arg2: !migraphx.shaped<1x5x3xf32, 15x3x1>, %arg3: !migraphx.shaped<1x5x3xf32, 15x3x1>) -> (!migraphx.shaped<1x5x3xf32, 15x3x1>, !migraphx.shaped<1x5x1xf32, 5x1x1>) attributes{arch = "", enable_splitk_for_tuning, kernel = "mixr"} {
    %0 = migraphx.dot %arg0, %arg1 {perf_config="v2:16,32,16,16,16,16,16,1,1"} : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1> -> <1x5x3xf32, 15x3x1>
    %cst = migraphx.literal(dense<3.000000e+00> : tensor<1x5x3xf32>) : <1x5x3xf32, 0x0x0>
    %1 = migraphx.add %arg2, %cst {} : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 0x0x0> -> <1x5x3xf32, 15x3x1>
    %2 = migraphx.mul %0, %1 {} : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1> -> <1x5x3xf32, 15x3x1>
    %3 = migraphx.add %0, %arg3 {} : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1> -> <1x5x3xf32, 15x3x1>
    %4 = migraphx.add %2, %3 {} : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1> -> <1x5x3xf32, 15x3x1>
    %5 = migraphx.reduce_sum %4 {axes = [2]} : <1x5x3xf32, 15x3x1> -> <1x5x1xf32, 5x1x1>
    return %4, %5 : !migraphx.shaped<1x5x3xf32, 15x3x1>, !migraphx.shaped<1x5x1xf32, 5x1x1>
  }
}
