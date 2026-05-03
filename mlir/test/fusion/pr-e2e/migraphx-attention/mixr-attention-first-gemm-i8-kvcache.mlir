// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -rand_min_int 32 -rand_max_int 64 -rand_type_int_for_inputs=3 -fut mlir_attention_wrapper -relDiff_threshold 0.000004 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

// Regression: i8 Q/K + kvcache mask used to crash the host decompose
// because mask -inf injection happens on QK in element type at mask
// time, and getNegInfAttr asserts on integer. The decompose now converts
// QK to softmaxType BEFORE masks (the verifier guarantees softmaxType is
// set whenever the value entering softmax doesn't already match V), so
// masks always operate on a float type.
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x64x32xi8, 2048x32x1>,
                                    %arg1: !migraphx.shaped<1x32x64xi8, 2048x64x1>,
                                    %arg2: !migraphx.shaped<1x64x32xf32, 2048x32x1>,
                                    %sl: !migraphx.shaped<1xi32, 1>,
                                    %qscale: !migraphx.shaped<1x64x64xf32, 4096x64x1>)
                                    -> !migraphx.shaped<1x64x32xf32, 2048x32x1> {
    %0 = migraphx.attention %arg0, %arg1, %arg2
      pre_softmax_inputs(%qscale : !migraphx.shaped<1x64x64xf32, 4096x64x1>)
      current_seq_len(%sl : !migraphx.shaped<1xi32, 1>) {
      ^bb0(%qk: !migraphx.shaped<1x64x64xi32, 4096x64x1>,
           %scale: !migraphx.shaped<1x64x64xf32, 4096x64x1>):
        %dq = migraphx.dequantizelinear %qk, %scale
          : <1x64x64xi32, 4096x64x1>, <1x64x64xf32, 4096x64x1>
          -> <1x64x64xf32, 4096x64x1>
        migraphx.yield %dq : !migraphx.shaped<1x64x64xf32, 4096x64x1>
      } softmax_type = f32 features = kvcache
      : <1x64x32xi8, 2048x32x1>, <1x32x64xi8, 2048x64x1>, <1x64x32xf32, 2048x32x1>
      -> <1x64x32xf32, 2048x32x1>
    return %0 : !migraphx.shaped<1x64x32xf32, 2048x32x1>
  }
}
