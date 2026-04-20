// RUN: rocmlir-gen -fut quantize_scale_fp8_ocp --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx-linalg,highlevel -host-pipeline=migraphx-linalg,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut quantize_scale_fp8_ocp_wrapper --verifier clone -relDiff_threshold 0.00001 - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch  | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

module {
  // CHECK: [1 1 1]
  func.func @quantize_scale_fp8_ocp(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<1x1x1x64xf32, 64x64x64x1>) -> !migraphx.shaped<1x112x112x64xf8E4M3FN, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale : <1x112x112x64xf32, 802816x7168x64x1>, <1x1x1x64xf32, 64x64x64x1> -> <1x112x112x64xf8E4M3FN, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf8E4M3FN, 802816x7168x64x1>
  }
}