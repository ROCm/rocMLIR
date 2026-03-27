// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// Asymmetric causal: seqQ=4 < seqK=16 (decode-style)
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x2x4x8xf16, 64x32x8x1>,
                                     %arg1: !migraphx.shaped<1x2x8x16xf16, 256x128x16x1>,
                                     %arg2: !migraphx.shaped<1x2x16x8xf16, 256x128x8x1>)
                                     -> !migraphx.shaped<1x2x4x8xf16, 64x32x8x1> {
    %0 = migraphx.attention %arg0, %arg1, %arg2 {
    } features = causal
      : <1x2x4x8xf16, 64x32x8x1>, <1x2x8x16xf16, 256x128x16x1>, <1x2x16x8xf16, 256x128x8x1>
      -> <1x2x4x8xf16, 64x32x8x1>
    return %0 : !migraphx.shaped<1x2x4x8xf16, 64x32x8x1>
  }
}
