// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.000004 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

// migraphx.attention variant of pr-e2e/attention/mixr-attention-padded-scale.mlir.
// Padded 3D shape with a single-input scale body (mul).
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x7x3xf32, 21x3x1>,
                                    %arg1: !migraphx.shaped<1x3x7xf32, 21x7x1>,
                                    %arg2: !migraphx.shaped<1x7x3xf32, 21x3x1>,
                                    %scale: !migraphx.shaped<1x7x7xf32, 49x7x1>)
                                    -> !migraphx.shaped<1x7x3xf32, 21x3x1> {
    %0 = migraphx.attention %arg0, %arg1, %arg2
      pre_softmax_inputs(%scale : !migraphx.shaped<1x7x7xf32, 49x7x1>) {
      ^bb0(%qk: !migraphx.shaped<1x7x7xf32, 49x7x1>,
           %s: !migraphx.shaped<1x7x7xf32, 49x7x1>):
        %scaled = migraphx.mul %qk, %s
          : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1>
          -> <1x7x7xf32, 49x7x1>
        migraphx.yield %scaled : !migraphx.shaped<1x7x7xf32, 49x7x1>
      }
      : <1x7x3xf32, 21x3x1>, <1x3x7xf32, 21x7x1>, <1x7x3xf32, 21x3x1>
      -> !migraphx.shaped<1x7x3xf32, 21x3x1>
    return %0 : !migraphx.shaped<1x7x3xf32, 21x3x1>
  }
}
