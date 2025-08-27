// RUN: rocmlir-gen -fut mlir_bwd_data_conv --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_bwd_data_conv_wrapper --verifier clone -relDiff_threshold 0.000002 - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_bwd_data_conv(
      %input: !migraphx.shaped<1x1x3x3xf32, 9x3x3x1>,
      %filter: !migraphx.shaped<1x1x3x3xf32, 9x3x3x1>
    ) -> !migraphx.shaped<1x1x5x5xf32, 25x25x5x1> {
    %result = migraphx.backwards_data_convolution
      %input, %filter
      {padding = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], group = 1}
      : !migraphx.shaped<1x1x3x3xf32, 9x3x3x1>, !migraphx.shaped<1x1x3x3xf32, 9x3x3x1> -> !migraphx.shaped<1x1x5x5xf32, 25x25x5x1>
    return %result : !migraphx.shaped<1x1x5x5xf32, 25x25x5x1>
  }
}
