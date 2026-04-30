// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_attention_wrapper -relDiff_threshold 0.000004  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

// migraphx.attention variant of pr-e2e/attention/mixr-attention-first-gemm-i8-padded.mlir.
// Padded shapes (head_dim=3, seq=7) exercise non-power-of-two dims through
// the kernel-side padding paths.
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x7x3xi8, 21x3x1>,
                                    %arg1: !migraphx.shaped<1x3x7xi8, 21x7x1>,
                                    %arg2: !migraphx.shaped<1x7x3xf32, 21x3x1>,
                                    %arg3: !migraphx.shaped<1x7x7xf32, 49x7x1>,
                                    %qscale: !migraphx.shaped<1x7x7xf32, 49x7x1>)
                                    -> (!migraphx.shaped<1x7x3xf32, 21x3x1>) {
    %0 = migraphx.attention %arg0, %arg1, %arg2
      pre_softmax_inputs(%qscale, %arg3
        : !migraphx.shaped<1x7x7xf32, 49x7x1>,
          !migraphx.shaped<1x7x7xf32, 49x7x1>) {
      ^bb0(%qk: !migraphx.shaped<1x7x7xi32, 49x7x1>,
           %scale: !migraphx.shaped<1x7x7xf32, 49x7x1>,
           %bias: !migraphx.shaped<1x7x7xf32, 49x7x1>):
        %dq = migraphx.dequantizelinear %qk, %scale
          : <1x7x7xi32, 49x7x1>, <1x7x7xf32, 49x7x1>
          -> <1x7x7xf32, 49x7x1>
        %biased = migraphx.add %dq, %bias
          : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1>
          -> <1x7x7xf32, 49x7x1>
        migraphx.yield %biased : !migraphx.shaped<1x7x7xf32, 49x7x1>
      } softmax_type = f32
      : <1x7x3xi8, 21x3x1>, <1x3x7xi8, 21x7x1>, <1x7x3xf32, 21x3x1>
      -> <1x7x3xf32, 21x3x1>
    return %0 : !migraphx.shaped<1x7x3xf32, 21x3x1>
  }
}
