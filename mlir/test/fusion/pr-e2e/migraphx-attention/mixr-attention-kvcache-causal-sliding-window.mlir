// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -rand_min_int 0 -rand_max_int 4 -rand_type_int_for_inputs=3 -fut mlir_attention_wrapper -RMS_threshold 0.02 -relDiff_threshold 0.1 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

// Thresholds note: this test runs softmax in f16 (no explicit
// softmax_type, so the lowering defaults to V's element type, f16) with
// currentSeqLen ranging [0, 4]. The combination of f16 softmax and the
// kvcache+causal+sliding-window mask chain produces enough dynamic
// range that the per-element relDiff vs. the f32 reference can hit
// ~0.05 even on this tiny shape (seqK=8, head_dim=2). 0.02 / 0.1 are
// the standard "f16 softmax + masking" envelope used by other f16
// kvcache tests in this directory.
//
// The companion mixr-attention-kvcache-causal-sliding-window-clamp
// test pins currentSeqLen < windowSize and runs at the standard tight
// 0.0005 thresholds; that one is the accuracy regression detector for
// the masking math itself (which must be bit-exact between host and
// GPU under those conditions), so a real bug in the mask lowering
// would not silently hide behind this test's looser thresholds.
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x2x1x2xf16, 4x2x2x1>,
                                     %arg1: !migraphx.shaped<1x2x2x8xf16, 32x16x8x1>,
                                     %arg2: !migraphx.shaped<1x2x8x2xf16, 32x16x2x1>,
                                     %arg3: !migraphx.shaped<1x2xi32, 2x1>)
                                     -> !migraphx.shaped<1x2x1x2xf16, 4x2x2x1> {
    %0 = migraphx.attention %arg0, %arg1, %arg2
      current_seq_len(%arg3 : !migraphx.shaped<1x2xi32, 2x1>) {
      } features = "kvcache|causal|sliding_window" slidingWindowSize = 4
      : <1x2x1x2xf16, 4x2x2x1>, <1x2x2x8xf16, 32x16x8x1>, <1x2x8x2xf16, 32x16x2x1>
      -> <1x2x1x2xf16, 4x2x2x1>
    return %0 : !migraphx.shaped<1x2x1x2xf16, 4x2x2x1>
  }
}
