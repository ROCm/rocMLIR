// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel | rocmlir-gen -ph -print-results -rand fixed - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
module {
    // CHECK : Unranked Memref base
  // CHECK:  [-0.5, -1, 0.75, -0.5]
  func.func @mlir_quantizelinear_quantizelinear_quant_dot_dequantizelinear(%arg0: !migraphx.shaped<2x2xf32, 2x1>, %arg1: !migraphx.shaped<2x2xf32, 2x1>) -> !migraphx.shaped<2x2xf32, 2x1> attributes {arch = "##TOKEN_ARCH##", kernel = "mixr"} {
    %0 = migraphx.literal(dense<2.500000e-01> : tensor<1xf32>) : <1xf32, 0>
    %1 = migraphx.literal(dense<5.000000e-01> : tensor<1xf32>) : <1xf32, 0>
    %2 = migraphx.literal(dense<5.000000e-01> : tensor<1xf32>) : <1xf32, 0>
    %3 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [2, 2]} : <1xf32, 0> -> <2x2xf32, 0x0>
    %4 = migraphx.quantizelinear %arg0, %3 {out_type = 13 : i64} : <2x2xf32, 2x1>, <2x2xf32, 0x0> -> <2x2xf8E4M3FN, 2x1>
    %5 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [2, 2]} : <1xf32, 0> -> <2x2xf32, 0x0>
    %6 = migraphx.quantizelinear %arg1, %5 {out_type = 13 : i64} : <2x2xf32, 2x1>, <2x2xf32, 0x0> -> <2x2xf8E4M3FN, 2x1>
    %7 = migraphx.quant_dot %4, %6 : <2x2xf8E4M3FN, 2x1>, <2x2xf8E4M3FN, 2x1> -> <2x2xf32, 2x1>
    %8 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [2, 2]} : <1xf32, 0> -> <2x2xf32, 0x0>
    %9 = migraphx.dequantizelinear %7, %8 : <2x2xf32, 2x1>, <2x2xf32, 0x0> -> <2x2xf32, 2x1>
    return %9 : !migraphx.shaped<2x2xf32, 2x1>
  }
}
