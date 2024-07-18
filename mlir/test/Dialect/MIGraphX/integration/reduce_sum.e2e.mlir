// RUN: rocmlir-gen --clone-harness -arch %arch -fut reduceSum %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -pr -verifier clone -rand=none -fut reduceSum_wrapper - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full --arch %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK
// ALLOW_RETRIES: 2
// CHECK: 4, 1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1

func.func @reduceSum(%arg0: !migraphx.shaped<1x4x4x4xf32, 64x16x4x1>) -> !migraphx.shaped<1x4x1x4xf32, 16x4x4x1> {
    %0 = migraphx.reduce_sum %arg0 {axes = [2 : i64]} : <1x4x4x4xf32, 64x16x4x1> -> <1x4x1x4xf32, 16x4x4x1>
    func.return %0 : !migraphx.shaped<1x4x1x4xf32, 16x4x4x1>
}
