// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -rand_min_int 0 -rand_max_int 64 -rand_type_int_for_inputs=3 -fut mlir_attention_wrapper -RMS_threshold 0.02 -relDiff_threshold 0.1 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

// Thresholds note: this test runs softmax in f16 (no explicit
// softmax_type, so the lowering defaults to V's element type, f16). f16
// has only ~3 decimal digits of mantissa precision, and reduce_sum +
// exp + recip in softmax are the worst part of the attention chain for
// dynamic range -- the empirical RMS error vs. the f32 reference for a
// seqK=128, head_dim=64 attention here is ~1e-2 and the per-element
// relDiff peaks around 0.05, so 0.02 / 0.1 give a small headroom.
// Compare with mixr-attention-kvcache-scale-lse.mlir, which exercises
// the same kvcache path with softmax_type = f32 and passes at the
// standard tight 0.0005 thresholds. Tightening these requires either
// adding softmax_type = f32 (changes what's tested) or accepting some
// flakiness in the loosest f16 cases. The masking math itself (causal,
// kvcache, sliding-window) is exercised at tight thresholds by the
// kvcache-causal-sliding-window-clamp test, so a regression there
// would not silently hide here.
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x2x1x64xf16, 128x64x64x1>,
                                     %arg1: !migraphx.shaped<1x2x64x128xf16, 16384x8192x128x1>,
                                     %arg2: !migraphx.shaped<1x2x128x64xf16, 16384x8192x64x1>,
                                     %arg3: !migraphx.shaped<1x2xi32, 2x1>)
                                     -> !migraphx.shaped<1x2x1x64xf16, 128x64x64x1> {
    %0 = migraphx.attention %arg0, %arg1, %arg2
      current_seq_len(%arg3 : !migraphx.shaped<1x2xi32, 2x1>) {
      } features = kvcache
      : <1x2x1x64xf16, 128x64x64x1>, <1x2x64x128xf16, 16384x8192x128x1>, <1x2x128x64xf16, 16384x8192x64x1>
      -> <1x2x1x64xf16, 128x64x64x1>
    return %0 : !migraphx.shaped<1x2x1x64xf16, 128x64x64x1>
  }
}
