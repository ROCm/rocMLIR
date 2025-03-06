// RUN: rocmlir-gen -fut dot_splitk_dequantizelinear --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand none -fut dot_splitk_dequantizelinear_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// ALLOW_RETRIES: 2
module {
  // CHECK: [1 1 1]
  // CHECK: [1 1 1]
  func.func @dot_splitk_dequantizelinear(%arg0: !migraphx.shaped<1x128x1x1xf32, 128x1x1x1>, %arg1: !migraphx.shaped<1x128x56x56xi8, 401408x3136x56x1>, %arg2: !migraphx.shaped<128x128x3x3xi8, 1152x9x3x1>) -> (!migraphx.shaped<1x128x28x28xf32, 100352x784x28x1>, !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>) attributes{arch = "", enable_splitk_for_tuning, kernel = "mixr"} {
    %1 = migraphx.quant_convolution %arg1, %arg2 {perf_config="v2:16,32,16,16,16,16,16,1,1", dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [2, 2]} : <1x128x56x56xi8, 401408x3136x56x1>, <128x128x3x3xi8, 1152x9x3x1> -> <1x128x28x28xi32, 100352x784x28x1>
    %2 = migraphx.dequantizelinear %1, %arg0 : <1x128x28x28xi32, 100352x784x28x1>, <1x128x1x1xf32, 128x1x1x1> -> <1x128x28x28xf32, 100352x784x28x1>
    %3 = migraphx.reduce_sum %2 {axes = [1]} : <1x128x28x28xf32, 100352x784x28x1> -> <1x1x28x28xf32, 784x784x28x1>
    return %2, %3 : !migraphx.shaped<1x128x28x28xf32, 100352x784x28x1>, !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>
  }
}
