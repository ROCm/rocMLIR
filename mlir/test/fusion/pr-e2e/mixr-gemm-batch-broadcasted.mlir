// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE

// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

module {
  func.func @test(%arg0: !migraphx.shaped<1x32x16xf16, 512x1x32>, %arg1: !migraphx.shaped<1x64x32xf16, 2048x1x64>, %arg2: !migraphx.shaped<4x32x2x4x4xf16, 0x2x1x256x64>) -> !migraphx.shaped<4x32x2x4x4xf16, 1024x2x1x256x64> {
    %trans0 = migraphx.transpose %arg0 {permutation = [0, 2, 1]} : <1x32x16xf16, 512x1x32> -> <1x16x32xf16, 512x32x1>
    %trans1 = migraphx.transpose %arg1 {permutation = [0, 2, 1]} : <1x64x32xf16, 2048x1x64> -> <1x32x64xf16, 2048x64x1>
    %0 = migraphx.multibroadcast %trans0 {out_dyn_dims = [], out_lens = [4, 16, 32]} : <1x16x32xf16, 512x32x1> -> <4x16x32xf16, 512x32x1>
    %1 = migraphx.multibroadcast %trans1 {out_dyn_dims = [], out_lens = [4, 32, 64]} : <1x32x64xf16, 2048x64x1> -> <4x32x64xf16, 2048x64x1>
    %2 = migraphx.dot %0, %1 : <4x16x32xf16, 512x32x1>, <4x32x64xf16, 2048x64x1> -> <4x16x64xf16, 1024x64x1>
    %3 = migraphx.reshape %2 {dims = [4, 4, 4, 32, 2]} : <4x16x64xf16, 1024x64x1> -> <4x4x4x32x2xf16, 1024x256x64x2x1>
    %4 = migraphx.transpose %3 {permutation = [0, 3, 4, 1, 2]} : <4x4x4x32x2xf16, 1024x256x64x2x1> -> <4x32x2x4x4xf16, 1024x2x1x256x64>
    %5 = migraphx.add %4, %arg2 : <4x32x2x4x4xf16, 1024x2x1x256x64>, <4x32x2x4x4xf16, 0x2x1x256x64> -> <4x32x2x4x4xf16, 1024x2x1x256x64>
    return %5 : !migraphx.shaped<4x32x2x4x4xf16, 1024x2x1x256x64>
  }
}
