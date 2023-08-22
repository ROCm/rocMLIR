// RUN: rocmlir-gen -fut mlir_slice_convolution_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_slice_convolution_add_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
func.func private @mlir_slice_convolution_add(%arg0: !migraphx.shaped<1x2x4x4xf32, 32x16x4x1>, %arg1: !migraphx.shaped<1x8x4x4xf32, 128x16x4x1>, %arg2: !migraphx.shaped<2x2x1x7xf32, 14x7x7x1>) -> !migraphx.shaped<1x2x4x4xf32, 32x16x4x1> {
    %0 = migraphx.slice %arg1 {axes = [1], ends = [4], starts = [2]} : !migraphx.shaped<1x8x4x4xf32, 128x16x4x1> -> !migraphx.shaped<1x2x4x4xf32, 32x16x4x1>
    %1 = migraphx.convolution %0, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [0, 3, 0, 3], padding_mode = 0 : i64, stride = [1, 1]} : !migraphx.shaped<1x2x4x4xf32, 32x16x4x1>, !migraphx.shaped<2x2x1x7xf32, 14x7x7x1> -> !migraphx.shaped<1x2x4x4xf32, 32x16x4x1>
    %2 = migraphx.add %1, %arg0 : !migraphx.shaped<1x2x4x4xf32, 32x16x4x1>, !migraphx.shaped<1x2x4x4xf32, 32x16x4x1> -> !migraphx.shaped<1x2x4x4xf32, 32x16x4x1>
    return %2 : !migraphx.shaped<1x2x4x4xf32, 32x16x4x1>
}
