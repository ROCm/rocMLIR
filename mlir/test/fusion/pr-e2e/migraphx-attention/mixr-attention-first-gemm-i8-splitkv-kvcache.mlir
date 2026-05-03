// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -rand_min_int 80 -rand_max_int 128 -rand_type_int_for_inputs=3 -fut mlir_attention_wrapper -RMS_threshold 0.0005 -relDiff_threshold 0.0005 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: [1 1 1]

// Cross-product coverage for the three independent feature flags that
// each individually have an E2E test in this directory:
//   * i8 Q/K (forces migraphx.quant_dot first GEMM, body dequantize)
//   * splitkv (split-space outputs, per-chunk LSE)
//   * kvcache (currentSeqLen masking)
//
// Individual tests:
//   - mixr-attention-first-gemm-i8-kvcache.mlir       (i8 + kvcache)
//   - mixr-attention-splitkv-kvcache.mlir             (splitkv + kvcache)
//   - mixr-attention-first-gemm-i8.mlir               (i8 alone)
//
// The combined path is the most complex single configuration the op
// supports: the host decompose has to (1) emit quant_dot with i8
// operands, (2) reshape K and broadcast Q for splitKV in QK-shape
// integer space, (3) inline a dequantize body that operates in
// split-space [B, splitKV, seqQ, seqK/splitKV], (4) apply the kvcache
// mask using global key indices that span splitKV chunks, and
// (5) compute per-chunk LSE in f32. The GPU side has to mirror that
// flow through gridwise_attention_accel with the splitKV transform
// inversion in postProcessFirstGemm. --verifier clone confirms the
// two paths agree to the standard tight 0.0005 thresholds.
//
// currentSeqLen is constrained to [80, 128) so every split chunk
// (seqK=128 / splitKV=2 = 64 keys per chunk) has at least 16 valid
// keys; a fully masked chunk would produce -inf per-chunk LSE and a
// per-chunk comparison failure even though the merged final result
// would still be correct (see splitkv-kvcache test for the same
// reasoning).
module {
  func.func private @mlir_attention(%q: !migraphx.shaped<1x64x32xi8, 2048x32x1>,
                                    %k: !migraphx.shaped<1x32x128xi8, 4096x128x1>,
                                    %v: !migraphx.shaped<1x128x32xf32, 4096x32x1>,
                                    %sl: !migraphx.shaped<1xi32, 1>,
                                    %scale: !migraphx.shaped<1x2x64x64xf32, 8192x4096x64x1>)
                                    -> (!migraphx.shaped<1x2x64x32xf32, 4096x2048x32x1>,
                                        !migraphx.shaped<1x2x64xf32, 128x64x1>) {
    %0, %1 = migraphx.attention %q, %k, %v
      pre_softmax_inputs(%scale : !migraphx.shaped<1x2x64x64xf32, 8192x4096x64x1>)
      current_seq_len(%sl : !migraphx.shaped<1xi32, 1>) {
      ^bb0(%qk: !migraphx.shaped<1x2x64x64xi32, 8192x4096x64x1>,
           %s: !migraphx.shaped<1x2x64x64xf32, 8192x4096x64x1>):
        %dq = migraphx.dequantizelinear %qk, %s
          : <1x2x64x64xi32, 8192x4096x64x1>, <1x2x64x64xf32, 8192x4096x64x1>
          -> <1x2x64x64xf32, 8192x4096x64x1>
        migraphx.yield %dq : !migraphx.shaped<1x2x64x64xf32, 8192x4096x64x1>
      } softmax_type = f32 features = "splitkv|kvcache" splitKV = 2
      : <1x64x32xi8, 2048x32x1>, <1x32x128xi8, 4096x128x1>, <1x128x32xf32, 4096x32x1>
      -> <1x2x64x32xf32, 4096x2048x32x1>, !migraphx.shaped<1x2x64xf32, 128x64x1>
    return %0, %1 : !migraphx.shaped<1x2x64x32xf32, 4096x2048x32x1>, !migraphx.shaped<1x2x64xf32, 128x64x1>
  }
}
