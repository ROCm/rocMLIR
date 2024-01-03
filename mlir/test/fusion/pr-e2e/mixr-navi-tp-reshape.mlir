// RUN: rocmlir-gen -fut mlir_transpose_reshape_dot_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel --verify-passes | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_transpose_reshape_dot_add_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
func.func @mlir_transpose_reshape_dot_add(%arg0: !migraphx.shaped<1x1001xf32, 0x1>, %arg1: !migraphx.shaped<1x1536x1x1xf32, 1536x1x1x1>, %arg2: !migraphx.shaped<1536x1001xf32, 1001x1>) -> !migraphx.shaped<1x1001xf32, 1001x1> {
    %0 = migraphx.transpose %arg1 {permutation = [0, 2, 3, 1]} : <1x1536x1x1xf32, 1536x1x1x1> -> <1x1x1x1536xf32, 1536x1x1x1>
    %1 = migraphx.reshape %0 {dims = [1, -1]} : <1x1x1x1536xf32, 1536x1x1x1> -> <1x1536xf32, 1536x1>
    %2 = migraphx.dot %1, %arg2 : <1x1536xf32, 1536x1>, <1536x1001xf32, 1001x1> -> <1x1001xf32, 1001x1>
    %3 = migraphx.add %2, %arg0 : <1x1001xf32, 1001x1>, <1x1001xf32, 0x1> -> <1x1001xf32, 1001x1>
    return %3 : !migraphx.shaped<1x1001xf32, 1001x1>
}
