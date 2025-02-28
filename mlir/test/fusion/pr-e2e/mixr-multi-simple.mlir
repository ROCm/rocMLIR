// RUN: rocmlir-gen -fut multi_simpl --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut multi_simpl_wrapper --verifier clone -| rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2

// CHECK: [1 1 1]
// CHECK: [1 1 1]
module {
  func.func @multi_simpl(%arg0: !migraphx.shaped<2x128x64x64xf16, 0x1x0x0>, %arg1: !migraphx.shaped<2x64x64x64xf16, 262144x1x4096x64>, %arg2: !migraphx.shaped<128x64x1x1xf16, 64x1x8192x8192>) -> (!migraphx.shaped<2x128x64x64xf16, 524288x1x8192x128>, !migraphx.shaped<2x128x64x64xf16, 524288x1x8192x128>)
  {
    %0 = migraphx.convolution %arg1, %arg2 {perf_config="v2:64,64,8,32,32,8,1,1,1", dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <2x64x64x64xf16, 262144x1x4096x64>, <128x64x1x1xf16, 64x1x8192x8192> -> <2x128x64x64xf16, 524288x1x8192x128>
    %1 = migraphx.add %0, %arg0 : <2x128x64x64xf16, 524288x1x8192x128>, <2x128x64x64xf16, 0x1x0x0> -> <2x128x64x64xf16, 524288x1x8192x128>
    return %0, %1 : !migraphx.shaped<2x128x64x64xf16, 524288x1x8192x128>, !migraphx.shaped<2x128x64x64xf16, 524288x1x8192x128>
  }
}
