// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -rand_min_int 0 -rand_max_int 2 -rand_type_int_for_inputs=3 -fut mlir_attention_wrapper -RMS_threshold 0.02 -relDiff_threshold 0.1 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

// Sliding-window clamp regression test.
//
// The sliding-window mask uses lowerBound = max(0, currentSeqLen - windowSize).
// Without the clamp, currentSeqLen < windowSize would yield a negative
// lowerBound and the mask would silently behave differently on host
// (signed wrap-around with APInt) vs. GPU (gridwise clamps via arith.maxsi).
//
// This test pins currentSeqLen to [0, 2] (rand_max_int = 2) while
// slidingWindowSize = 4, so the unclamped formula always produces a
// negative lowerBound. With the clamp on both paths, lowerBound = 0
// for every batch and the sliding-window mask is a no-op (only the
// causal + kvcache masks fire). --verifier clone catches any divergence
// between host and GPU.
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
