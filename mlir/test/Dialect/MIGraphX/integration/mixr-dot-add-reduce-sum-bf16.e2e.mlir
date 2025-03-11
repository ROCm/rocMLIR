// RUN: rocmlir-gen -fut dot_add_reduce_sum_bf16 --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut dot_add_reduce_sum_bf16_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
func.func @dot_add_reduce_sum_bf16(%arg0: !migraphx.shaped<1x5x4xbf16, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xbf16, 12x3x1>, %arg2: !migraphx.shaped<1x5x3xbf16, 15x3x1>) -> !migraphx.shaped<1x5x1xbf16, 5x1x1> {
    %0 = migraphx.dot %arg0, %arg1 : <1x5x4xbf16, 20x4x1>, <1x4x3xbf16, 12x3x1> -> <1x5x3xbf16, 15x3x1>
    %1 = migraphx.add %0, %arg2 {} : <1x5x3xbf16, 15x3x1>, <1x5x3xbf16, 15x3x1> -> <1x5x3xbf16, 15x3x1>
    %2 = migraphx.reduce_sum %1 {axes = [2 : i64]} : <1x5x3xbf16, 15x3x1> -> <1x5x1xbf16, 5x1x1>
    return %2 : !migraphx.shaped<1x5x1xbf16, 5x1x1>
}
