// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

module {
  func.func @test(%arg0: !migraphx.shaped<1x64x56x56xf32, 0x1x0x0>, %arg1: !migraphx.shaped<1x256x56x56xf32, 802816x3136x56x1>, %arg2: !migraphx.shaped<64x256x1x1xf32, 256x1x1x1>) -> !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1> {
    %2 = migraphx.add %1, %arg0 : <1x64x56x56xf32, 200704x3136x56x1>, <1x64x56x56xf32, 0x1x0x0> -> <1x64x56x56xf32, 200704x3136x56x1>
    %3 = migraphx.relu %2 : <1x64x56x56xf32, 200704x3136x56x1> -> <1x64x56x56xf32, 200704x3136x56x1>
    return %3 : !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1>
  }
}
