// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.000004 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

// migraphx.attention variant of pr-e2e/attention/mixr-attention-gqa-scale.mlir.
// 4D GQA (numHeadsQ=4, numHeadsKV=2) with a per-element scale applied in
// the preSoftmaxBody via migraphx.mul.
module {
  func.func private @mlir_attention(%q: !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>,
                                    %k: !migraphx.shaped<2x2x32x32xf32, 2048x1024x32x1>,
                                    %v: !migraphx.shaped<2x2x32x32xf32, 2048x1024x32x1>,
                                    %scale: !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>)
                                    -> (!migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>) {
    %0 = migraphx.attention %q, %k, %v
      pre_softmax_inputs(%scale : !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>) {
      ^bb0(%qk: !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>,
           %s: !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>):
        %scaled = migraphx.mul %qk, %s
          : <2x4x32x32xf32, 4096x1024x32x1>, <2x4x32x32xf32, 4096x1024x32x1>
          -> <2x4x32x32xf32, 4096x1024x32x1>
        migraphx.yield %scaled : !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>
      }
      : <2x4x32x32xf32, 4096x1024x32x1>, <2x2x32x32xf32, 2048x1024x32x1>, <2x2x32x32xf32, 2048x1024x32x1>
      -> <2x4x32x32xf32, 4096x1024x32x1>
    return %0 : !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>
  }
}
