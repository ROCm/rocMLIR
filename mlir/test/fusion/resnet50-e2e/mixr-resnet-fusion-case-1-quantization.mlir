// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

module {
  func.func @test(%arg0: !migraphx.shaped<1x128x1x1xf32, 128x1x1x1>, %arg1: !migraphx.shaped<1x128x56x56xi8, 401408x3136x56x1>, %arg2: !migraphx.shaped<128x128x3x3xi8, 1152x9x3x1>) -> !migraphx.shaped<1x128x28x28xi8, 100352x784x28x1> {
    %2 = migraphx.dequantizelinear %1, %arg0 : <1x128x28x28xi32, 100352x784x28x1>, <1x128x1x1xf32, 128x1x1x1> -> <1x128x28x28xf32, 100352x784x28x1>
    %3 = migraphx.quantizelinear %2, %arg0 : <1x128x28x28xf32, 100352x784x28x1>, <1x128x1x1xf32, 128x1x1x1> -> <1x128x28x28xi8, 100352x784x28x1>
    return %3 : !migraphx.shaped<1x128x28x28xi8, 100352x784x28x1>
  }
}

