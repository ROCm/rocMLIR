// RUN: rocmlir-gen -fut mlir_convolution_multi_output_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_convolution_multi_output_add_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// We need a check for each output as this test case has two outputs in it.
// CHECK: [1 1 1]
// CHECK: [1 1 1]
module {
  func.func @mlir_convolution_multi_output_add(%arg0: !migraphx.shaped<2x4x64x64xf16, 16384x4096x64x1>, %arg1: !migraphx.shaped<320x4x3x3xf16, 36x9x3x1>) -> (!migraphx.shaped<2x320x64x64xf16, 1310720x4096x64x1>, !migraphx.shaped<2x320x64x64xf16, 1310720x4096x64x1>) attributes{arch = "", enable_splitk_for_tuning, kernel = "mixr"} {
    %1 = migraphx.literal(dense<2.44140629E-5> : tensor<1xf16>) : <1xf16, 0>
    %2 = migraphx.literal(dense<1.0> : tensor<1xf16>) : <1xf16, 0>
    %3 = migraphx.literal(dense<2.0> : tensor<1xf16>) : <1xf16, 0>
    %4 = migraphx.convolution %arg0, %arg1 {perf_config="v3:16,32,4,16,16,4,4,1,2,1,1", dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <2x4x64x64xf16, 16384x4096x64x1>, <320x4x3x3xf16, 36x9x3x1> -> <2x320x64x64xf16, 1310720x4096x64x1>
    %5 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [2, 320, 64, 64]} : <1xf16, 0> -> <2x320x64x64xf16, 0x0x0x0>
    %6 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [2, 320, 64, 64]} : <1xf16, 0> -> <2x320x64x64xf16, 0x0x0x0>
    %7 = migraphx.multibroadcast %3 {out_dyn_dims = [], out_lens = [2, 320, 64, 64]} : <1xf16, 0> -> <2x320x64x64xf16, 0x0x0x0>
    %8 = migraphx.mul %4, %5 : <2x320x64x64xf16, 1310720x4096x64x1>, <2x320x64x64xf16, 0x0x0x0> -> <2x320x64x64xf16, 1310720x4096x64x1>
    %9 = migraphx.add %8, %6 : <2x320x64x64xf16, 1310720x4096x64x1>, <2x320x64x64xf16, 0x0x0x0> -> <2x320x64x64xf16, 1310720x4096x64x1>
    %10 = migraphx.add %8, %7 : <2x320x64x64xf16, 1310720x4096x64x1>, <2x320x64x64xf16, 0x0x0x0> -> <2x320x64x64xf16, 1310720x4096x64x1>
    return %9, %10 : !migraphx.shaped<2x320x64x64xf16, 1310720x4096x64x1>, !migraphx.shaped<2x320x64x64xf16, 1310720x4096x64x1>
  }
}
