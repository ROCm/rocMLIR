// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -RMS_threshold 0.0005 -relDiff_threshold 0.0005 -fut mlir_attention_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: [1 1 1]
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x4x64x64xf16, 16384x4096x64x1>,
                                     %arg1: !migraphx.shaped<1x4x64x128xf16, 32768x8192x128x1>,
                                     %arg2: !migraphx.shaped<1x4x128x64xf16, 32768x8192x64x1>)
                                     -> (!migraphx.shaped<1x4x2x64x64xf16, 32768x8192x4096x64x1>,
                                         !migraphx.shaped<1x4x2x64xf32, 512x128x64x1>) {
    %0, %1 = migraphx.attention %arg0, %arg1, %arg2 {
    } softmax_type = f32 features = splitkv splitKV = 2
      : <1x4x64x64xf16, 16384x4096x64x1>, <1x4x64x128xf16, 32768x8192x128x1>, <1x4x128x64xf16, 32768x8192x64x1>
      -> <1x4x2x64x64xf16, 32768x8192x4096x64x1>, !migraphx.shaped<1x4x2x64xf32, 512x128x64x1>
    return %0, %1 : !migraphx.shaped<1x4x2x64x64xf16, 32768x8192x4096x64x1>, !migraphx.shaped<1x4x2x64xf32, 512x128x64x1>
  }
}
