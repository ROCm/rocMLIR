// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

module {
  func.func @test(%arg0: !migraphx.shaped<1x256x14x14xf32, 0x1x0x0>, %arg1: !migraphx.shaped<1x256x28x28xf32, 200704x784x28x1>, %arg2: !migraphx.shaped<256x256x3x3xf32, 2304x9x3x1>) -> !migraphx.shaped<1x256x14x14xf32, 50176x196x14x1> {
    %1 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [2, 2]} : !migraphx.shaped<1x256x28x28xf32, 200704x784x28x1>, !migraphx.shaped<256x256x3x3xf32, 2304x9x3x1> -> !migraphx.shaped<1x256x14x14xf32, 50176x196x14x1>
    %2 = migraphx.add %1, %arg0 : !migraphx.shaped<1x256x14x14xf32, 50176x196x14x1>, !migraphx.shaped<1x256x14x14xf32, 0x1x0x0> -> !migraphx.shaped<1x256x14x14xf32, 50176x196x14x1>
    %3 = migraphx.relu %2 : !migraphx.shaped<1x256x14x14xf32, 50176x196x14x1> -> !migraphx.shaped<1x256x14x14xf32, 50176x196x14x1>
    return %3 : !migraphx.shaped<1x256x14x14xf32, 50176x196x14x1>
  }
}
