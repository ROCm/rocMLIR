// RUN: rocmlir-gen -fut dot_splitk_add_ext --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand none -fut dot_splitk_add_ext_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut dot_splitk_add_ext --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -RMS_threshold=1e-3 -relDiff_threshold=1e-3 -ph -print-results -rand 1 -rand_type float -fut dot_splitk_add_ext_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
module {
  // CHECK: [1 1 1]
  // CHECK:  [5,     5,     5,  5,     5,     5,  5,     5,     5,  5,     5,     5,  5,     5,     5]

  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base

  func.func @dot_splitk_add_ext(%arg0: !migraphx.shaped<1x5x4xf16, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf16, 12x3x1>, %arg2: !migraphx.shaped<1x5x3xf16, 15x3x1>) -> !migraphx.shaped<1x5x3xf32, 15x3x1> attributes{arch = "", enable_splitk_for_tuning, kernel = "mixr"} {
    %0 = migraphx.dot %arg0, %arg1 {perf_config="v3:16,32,4,16,16,4,4,1,2,1,1"} : <1x5x4xf16, 20x4x1>, <1x4x3xf16, 12x3x1> -> <1x5x3xf16, 15x3x1>
    %2 = migraphx.add %0, %arg2 {} : <1x5x3xf16, 15x3x1>, <1x5x3xf16, 15x3x1> -> <1x5x3xf16, 15x3x1>
    %3 = migraphx.convert %2 : <1x5x3xf16, 15x3x1> to <1x5x3xf32, 15x3x1>
    return %3 : !migraphx.shaped<1x5x3xf32, 15x3x1>
  }
}
