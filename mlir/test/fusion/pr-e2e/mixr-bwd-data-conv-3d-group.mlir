// RUN: rocmlir-gen -fut mlir_bwd_data_conv --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx-linalg,highlevel -host-pipeline=migraphx-linalg,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_bwd_data_conv_wrapper --verifier clone -relDiff_threshold 0.00001 - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch  | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]
func.func @mlir_bwd_data_conv(
    %arg0: !migraphx.shaped<1x2x3x5x5xf32, 150x75x25x5x1>,
    %arg1: !migraphx.shaped<2x1x3x3x3xf32, 27x27x9x3x1>
) -> !migraphx.shaped<1x2x5x13x17xf32, 2210x1105x221x17x1> {
    %0 = migraphx.backwards_data_convolution %arg0, %arg1 {
        dilation = [2, 3, 4],
        group = 2 : i64,
        padding = [2, 3, 4, 2, 3, 4],
        padding_mode = 0 : i64,
        stride = [2, 3, 4]
    } : <1x2x3x5x5xf32, 150x75x25x5x1>, <2x1x3x3x3xf32, 27x27x9x3x1> -> <1x2x5x13x17xf32, 2210x1105x221x17x1>
    return %0 : !migraphx.shaped<1x2x5x13x17xf32, 2210x1105x221x17x1>
}