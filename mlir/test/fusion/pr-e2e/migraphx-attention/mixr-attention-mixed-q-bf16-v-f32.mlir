// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -RMS_threshold 0.005 -relDiff_threshold 0.05 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

// Q/K in bf16 with V in f32. softmaxType = f32 must be set explicitly per
// the verifier rule (the value entering softmax doesn't match V.elemType).
// Both host and GPU should run softmax in f32 and the second GEMM in V's
// element type (f32). End-to-end --verifier clone confirms the host CPU
// reference and GPU kernel agree.
//
// Thresholds note: relDiff_threshold = 0.05 is looser than the typical
// 0.0005 used for f16/f32 attention E2Es because of two cumulative
// sources of bf16 error:
//   1. The mfma path matmuls bf16 Q*K^T but the *softmax exp/sum* runs
//      in f32 on both sides, so the first-GEMM error caps at bf16's ~3
//      decimal digits of mantissa precision.
//   2. The host reference's second GEMM uses migraphx.dot, which
//      accumulates in the *operand* element type (f32 here once softmax
//      lifts to f32) -- not promoted from softmax's f32 the same way
//      the GPU mfma path widens. For long sequences this widens the
//      host/GPU gap further (see the dc0ddb34e08b commit message). For
//      this small 32x64 case the empirical worst-case relDiff is ~0.04
//      and RMS ~0.003; the 0.05/0.005 thresholds give modest headroom.
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
