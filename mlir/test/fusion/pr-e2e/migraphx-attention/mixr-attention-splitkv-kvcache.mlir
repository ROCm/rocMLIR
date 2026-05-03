// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -rand_min_int 80 -rand_max_int 128 -rand_type_int_for_inputs=3 -fut mlir_attention_wrapper -RMS_threshold 0.0005 -relDiff_threshold 0.0005 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: [1 1 1]

// splitKV + kvcache: KV-cache-decode shape (seqQ=1, large seqK) with
// splitKV=2 split-K work distribution. currentSeqLen is randomized via
// rand_type_int_for_inputs to exercise the per-batch kvcache mask
// within each split chunk; --verifier clone catches any divergence
// between the host decompose and the GPU rock.attention path. This is
// the GPU companion to host coverage in
// MIGraphXAttentionDecompose/attention-decompose.mlir.
//
// currentSeqLen is constrained to [80, 128) so every split chunk
// (seqK=128 / splitKV=2 = 64 keys per chunk) has at least 16 valid
// keys: the result tensor is split-space (5D), so the verifier
// compares per-chunk partial outputs and per-chunk LSE values
// directly. A fully masked chunk would produce -inf LSE / undefined
// partial output and the per-chunk comparison would fail even though
// the merged final result would still be correct.
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x2x1x64xf16, 128x64x64x1>,
                                     %arg1: !migraphx.shaped<1x2x64x128xf16, 16384x8192x128x1>,
                                     %arg2: !migraphx.shaped<1x2x128x64xf16, 16384x8192x64x1>,
                                     %arg3: !migraphx.shaped<1x2xi32, 2x1>)
                                     -> (!migraphx.shaped<1x2x2x1x64xf16, 256x128x64x64x1>,
                                         !migraphx.shaped<1x2x2x1xf32, 4x2x1x1>) {
    %0, %1 = migraphx.attention %arg0, %arg1, %arg2
      current_seq_len(%arg3 : !migraphx.shaped<1x2xi32, 2x1>) {
      } softmax_type = f32 features = "splitkv|kvcache" splitKV = 2
      : <1x2x1x64xf16, 128x64x64x1>, <1x2x64x128xf16, 16384x8192x128x1>, <1x2x128x64xf16, 16384x8192x64x1>
      -> <1x2x2x1x64xf16, 256x128x64x64x1>, !migraphx.shaped<1x2x2x1xf32, 4x2x1x1>
    return %0, %1 : !migraphx.shaped<1x2x2x1x64xf16, 256x128x64x64x1>, !migraphx.shaped<1x2x2x1xf32, 4x2x1x1>
  }
}
