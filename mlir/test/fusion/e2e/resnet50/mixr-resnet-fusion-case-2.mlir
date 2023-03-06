// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut test - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

module {
  func.func @test(%arg0: tensor<1x256x1x1xf32>, %arg1: tensor<1x256x28x28xf32>, %arg2: tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32> {
    %0 = migraphx.multibroadcast(%arg0) {out_lens = [1, 256, 14, 14]} : (tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %1 = migraphx.convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [2, 2]} : (tensor<1x256x28x28xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %2 = migraphx.add(%1, %0) : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %3 = migraphx.relu(%2) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    return %3 : tensor<1x256x14x14xf32>
  }
}
