// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-driver -kernel-pipeline migraphx %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut quant_dot --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2

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
