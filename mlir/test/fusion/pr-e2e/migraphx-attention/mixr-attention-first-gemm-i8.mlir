// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_attention_wrapper -relDiff_threshold 0.000004 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

// migraphx.attention variant of pr-e2e/attention/mixr-attention-first-gemm-i8.mlir.
// Same numerics: i8 Q/K, f32 V. The dequant scale and bias are full-shape
// f32 tensors handed directly into preSoftmaxBody via preSoftmaxElemWiseInputs.
// The original test had a per-tensor (1x1x1) scale; the migraphx.attention
// path requires preSoftmaxElemWiseInputs to match the QK shape, so the
// producer must materialise the broadcast before the op (omitted here for
// test-data simplicity by passing a full-shape scale operand directly).
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x64x32xi8, 2048x32x1>,
                                    %arg1: !migraphx.shaped<1x32x64xi8, 2048x64x1>,
                                    %arg2: !migraphx.shaped<1x64x32xf32, 2048x32x1>,
                                    %arg3: !migraphx.shaped<1x64x64xf32, 4096x64x1>,
                                    %qscale: !migraphx.shaped<1x64x64xf32, 4096x64x1>)
                                    -> (!migraphx.shaped<1x64x32xf32, 2048x32x1>) {
    %0 = migraphx.attention %arg0, %arg1, %arg2
      pre_softmax_inputs(%qscale, %arg3
        : !migraphx.shaped<1x64x64xf32, 4096x64x1>,
          !migraphx.shaped<1x64x64xf32, 4096x64x1>) {
      ^bb0(%qk: !migraphx.shaped<1x64x64xi32, 4096x64x1>,
           %scale: !migraphx.shaped<1x64x64xf32, 4096x64x1>,
           %bias: !migraphx.shaped<1x64x64xf32, 4096x64x1>):
        %dq = migraphx.dequantizelinear %qk, %scale
          : <1x64x64xi32, 4096x64x1>, <1x64x64xf32, 4096x64x1>
          -> <1x64x64xf32, 4096x64x1>
        %biased = migraphx.add %dq, %bias
          : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1>
          -> <1x64x64xf32, 4096x64x1>
        migraphx.yield %biased : !migraphx.shaped<1x64x64xf32, 4096x64x1>
      } softmax_type = f32
      : <1x64x32xi8, 2048x32x1>, <1x32x64xi8, 2048x64x1>, <1x64x32xf32, 2048x32x1>
      -> <1x64x32xf32, 2048x32x1>
    return %0 : !migraphx.shaped<1x64x32xf32, 2048x32x1>
  }
}
