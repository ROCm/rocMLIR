  // RUN: rocmlir-driver -kernel-pipeline migraphx %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -print-results -fut func_mbcast -verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2

module {
  // CHECK:  [1 1 1]

  func.func @func_mbcast(%arg0: !migraphx.shaped<1x64x1x1xf32, 64x1x1x1>, %arg1: !migraphx.shaped<1x3x224x224xf32, 150528x50176x224x1>, %arg2: !migraphx.shaped<64x3x7x7xf32, 147x49x7x1>) -> !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1> {
    %0 = migraphx.multibroadcast %arg0 {out_lens = [1, 64, 112, 112]} : <1x64x1x1xf32, 64x1x1x1> -> <1x64x112x112xf32, 64x1x0x0>
    %2 = migraphx.add %1, %0 : <1x64x112x112xf32, 802816x12544x112x1>, <1x64x112x112xf32, 64x1x0x0> -> <1x64x112x112xf32, 802816x12544x112x1>
    %3 = migraphx.relu %2 : <1x64x112x112xf32, 802816x12544x112x1> -> <1x64x112x112xf32, 802816x12544x112x1>
    return %3 : !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>
  }
}
