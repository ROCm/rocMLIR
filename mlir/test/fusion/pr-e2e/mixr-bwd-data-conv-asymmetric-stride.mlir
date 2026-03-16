// RUN: rocmlir-gen -fut mlir_bwd_data_conv --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_bwd_data_conv_wrapper --verifier clone -relDiff_threshold 0.00001 - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch  | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut mlir_bwd_data_conv --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx-linalg,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_bwd_data_conv_wrapper --verifier clone -relDiff_threshold 0.00001 - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch  | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut mlir_bwd_data_conv --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx-linalg,highlevel -host-pipeline=migraphx,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_bwd_data_conv_wrapper --verifier clone -relDiff_threshold 0.00001 - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch  | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

module {
  func.func @mlir_bwd_data_conv(
      %arg0: !migraphx.shaped<1x3x6x7xf32, 126x42x7x1>,
      %arg1: !migraphx.shaped<3x4x3x3xf32, 36x9x3x1>
  ) -> !migraphx.shaped<1x4x11x19xf32, 836x209x19x1> {
    // CHECK: [1 1 1]
    %0 = migraphx.backwards_data_convolution %arg0, %arg1 {
      dilation = [1, 1],
      group = 1 : i64,
      padding = [1, 1, 1, 1],
      padding_mode = 0 : i64,
      stride = [2, 3]} : <1x3x6x7xf32, 126x42x7x1>, <3x4x3x3xf32, 36x9x3x1> -> <1x4x11x19xf32, 836x209x19x1>
    return %0 : !migraphx.shaped<1x4x11x19xf32, 836x209x19x1>
  }
}
