// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

module {
  func.func @test(%arg0: !migraphx.shaped<1x512x1x1xf32, 512x1x1x1>, %arg1: !migraphx.shaped<1x512x14x14xf32, 100352x196x14x1>, %arg2: !migraphx.shaped<512x512x3x3xf32, 4608x9x3x1>) -> !migraphx.shaped<1x512x7x7xf32, 25088x49x7x1> {
    %0 = migraphx.multibroadcast %arg0 {out_lens = [1, 512, 7, 7]} : !migraphx.shaped<1x512x1x1xf32, 512x1x1x1> -> !migraphx.shaped<1x512x7x7xf32, 25088x49x7x1>
    %1 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [2, 2]} : !migraphx.shaped<1x512x14x14xf32, 100352x196x14x1>, !migraphx.shaped<512x512x3x3xf32, 4608x9x3x1> -> !migraphx.shaped<1x512x7x7xf32, 25088x49x7x1>
    %2 = migraphx.add %1, %0 : !migraphx.shaped<1x512x7x7xf32, 25088x49x7x1>, !migraphx.shaped<1x512x7x7xf32, 25088x49x7x1> -> !migraphx.shaped<1x512x7x7xf32, 25088x49x7x1>
    %3 = migraphx.relu %2 : !migraphx.shaped<1x512x7x7xf32, 25088x49x7x1> -> !migraphx.shaped<1x512x7x7xf32, 25088x49x7x1>
    return %3 : !migraphx.shaped<1x512x7x7xf32, 25088x49x7x1>
  }
}
