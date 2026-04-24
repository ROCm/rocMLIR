// RUN: rocmlir-gen -fut mlir_quantizelinear_f8E4M3FN --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx-linalg,highlevel -host-pipeline=migraphx,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_quantizelinear_f8E4M3FN_wrapper --verifier clone -relDiff_threshold 0.00001 - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch  | rocmlir-opt --emulate-fp8-ext-trunc | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]
func.func @mlir_quantizelinear_f8E4M3FN(%input: !migraphx.shaped<2x2xf32, 2x1>, %scale: !migraphx.shaped<2x2xf32, 2x1>, %bias: !migraphx.shaped<2x2xf8E4M3FN, 2x1>) -> !migraphx.shaped<2x2xf32, 2x1> {
    %result = migraphx.quantizelinear %input, %scale, %bias : <2x2xf32, 2x1>, <2x2xf32, 2x1>, !migraphx.shaped<2x2xf8E4M3FN, 2x1> -> <2x2xf8E4M3FN, 2x1>
    %dot_result = migraphx.quant_dot %result, %result : <2x2xf8E4M3FN, 2x1>, <2x2xf8E4M3FN, 2x1> -> <2x2xf32, 2x1>
    return %dot_result : !migraphx.shaped<2x2xf32, 2x1>
}
