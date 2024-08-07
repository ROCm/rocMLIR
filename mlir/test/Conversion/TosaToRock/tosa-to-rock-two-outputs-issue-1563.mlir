// RUN: rocmlir-gen -fut mlir_dot_transpose_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_dot_transpose_add_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
// CHECK: [1 1 1]
module {
  func.func @mlir_dot_transpose_add(%arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>, %arg1: !migraphx.shaped<1x4x5xf32, 20x5x1>, %arg2: !migraphx.shaped<1x5x5xf32, 25x5x1>) -> (!migraphx.shaped<1x4x5xf32, 20x5x1>, !migraphx.shaped<1x5x4xf32, 20x4x1>) attributes {} {
    %0 = migraphx.dot %arg1, %arg2 : <1x4x5xf32, 20x5x1>, <1x5x5xf32, 25x5x1> -> <1x4x5xf32, 20x5x1>
    %1 = migraphx.transpose %0 {permutation = [0, 2, 1]} : <1x4x5xf32, 20x5x1> -> <1x5x4xf32, 20x1x5>
    %2 = migraphx.add %1, %arg0 : <1x5x4xf32, 20x1x5>, <1x5x4xf32, 20x4x1> -> <1x5x4xf32, 20x4x1>
    return %0, %2 : !migraphx.shaped<1x4x5xf32, 20x5x1>, !migraphx.shaped<1x5x4xf32, 20x4x1>
  }
}
