// RUN: rocmlir-gen -fut dot_acc_f16 --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut dot_acc_f16_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @dot_acc_f16(%arg0: !migraphx.shaped<8x64x64x320xf16, 1310720x20480x320x1>, %arg1: !migraphx.shaped<8x64x320x320xf16, 6553600x102400x320x1>) -> !migraphx.shaped<8x64x64x320xf16, 1310720x20480x320x1>  attributes {kernel = "mixr"} {
    %4 = migraphx.dot %arg0, %arg1 {perf_config = "v2:16,16,8,16,16,4,1,1,1"} : <8x64x64x320xf16, 1310720x20480x320x1>, <8x64x320x320xf16, 6553600x102400x320x1> -> <8x64x64x320xf16, 1310720x20480x320x1>
    return %4 : !migraphx.shaped<8x64x64x320xf16, 1310720x20480x320x1>
  }
}
