// RUN: rocmlir-gen -fut quant_dot --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand none -fut quant_dot_wrapper - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut quant_dot --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut quant_dot_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE

module {
  // CHECK:  [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  func.func @quant_dot(%arg0: !migraphx.shaped<1x5x4xi8, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xi8, 12x3x1>) -> !migraphx.shaped<1x15xi32, 15x1> attributes{kernel, arch = ""} {
    %0 = migraphx.quant_dot %arg0, %arg1 : <1x5x4xi8, 20x4x1>, <1x4x3xi8, 12x3x1> -> <1x5x3xi32, 15x3x1>
    %1 = migraphx.reshape %0 {dims = [1:i64, 15:i64]} : <1x5x3xi32, 15x3x1> -> <1x15xi32, 15x1>
    return %1 : !migraphx.shaped<1x15xi32, 15x1>
  }
}
