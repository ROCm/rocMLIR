// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -RMS_threshold 0.01 -relDiff_threshold 0.05 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

// Q/K in bf16 with V in f32. softmaxType = f32 must be set explicitly per
// the verifier rule (the value entering softmax doesn't match V.elemType).
// Both host and GPU should run softmax in f32 and the second GEMM in V's
// element type (f32). End-to-end --verifier clone confirms the host CPU
// reference and GPU kernel agree.
module {
  func.func private @mlir_attention(%q: !migraphx.shaped<1x4x32x64xbf16, 8192x2048x64x1>,
                                    %k: !migraphx.shaped<1x4x64x32xbf16, 8192x2048x32x1>,
                                    %v: !migraphx.shaped<1x4x32x64xf32, 8192x2048x64x1>)
                                    -> !migraphx.shaped<1x4x32x64xf32, 8192x2048x64x1> {
    %0 = migraphx.attention %q, %k, %v {
    } softmax_type = f32
      : <1x4x32x64xbf16, 8192x2048x64x1>, <1x4x64x32xbf16, 8192x2048x32x1>, <1x4x32x64xf32, 8192x2048x64x1>
      -> <1x4x32x64xf32, 8192x2048x64x1>
    return %0 : !migraphx.shaped<1x4x32x64xf32, 8192x2048x64x1>
  }
}
