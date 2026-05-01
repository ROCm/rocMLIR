// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.000004 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

// migraphx.attention variant of
// pr-e2e/attention/mixr-attention-padded-complex-tree-elemwise.mlir.
// Multi-op tree body: aux inputs are combined leaf-level (sub, add), then
// the QK output is scaled and biased. Exercises a 4-op body with 4
// preSoftmaxElemWiseInputs.
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x7x3xf32, 21x3x1>,
                                    %arg1: !migraphx.shaped<1x3x7xf32, 21x7x1>,
                                    %arg2: !migraphx.shaped<1x7x3xf32, 21x3x1>,
                                    %arg3: !migraphx.shaped<1x7x7xf32, 49x7x1>,
                                    %arg4: !migraphx.shaped<1x7x7xf32, 49x7x1>,
                                    %arg5: !migraphx.shaped<1x7x7xf32, 49x7x1>,
                                    %arg6: !migraphx.shaped<1x7x7xf32, 49x7x1>)
                                    -> !migraphx.shaped<1x7x3xf32, 21x3x1> {
    %0 = migraphx.attention %arg0, %arg1, %arg2
      pre_softmax_inputs(%arg3, %arg4, %arg5, %arg6
        : !migraphx.shaped<1x7x7xf32, 49x7x1>,
          !migraphx.shaped<1x7x7xf32, 49x7x1>,
          !migraphx.shaped<1x7x7xf32, 49x7x1>,
          !migraphx.shaped<1x7x7xf32, 49x7x1>) {
      ^bb0(%qk: !migraphx.shaped<1x7x7xf32, 49x7x1>,
           %a3: !migraphx.shaped<1x7x7xf32, 49x7x1>,
           %a4: !migraphx.shaped<1x7x7xf32, 49x7x1>,
           %a5: !migraphx.shaped<1x7x7xf32, 49x7x1>,
           %a6: !migraphx.shaped<1x7x7xf32, 49x7x1>):
        // Leaf-level elemwise on auxiliary inputs.
        %sub = migraphx.sub %a3, %a4
          : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1>
          -> <1x7x7xf32, 49x7x1>
        %add = migraphx.add %a5, %a6
          : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1>
          -> <1x7x7xf32, 49x7x1>
        // Second-level: scale qk by sub, then bias by add.
        %scaled = migraphx.mul %qk, %sub
          : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1>
          -> <1x7x7xf32, 49x7x1>
        %biased = migraphx.add %scaled, %add
          : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1>
          -> <1x7x7xf32, 49x7x1>
        migraphx.yield %biased : !migraphx.shaped<1x7x7xf32, 49x7x1>
      }
      : <1x7x3xf32, 21x3x1>, <1x3x7xf32, 21x7x1>, <1x7x3xf32, 21x3x1>
      -> !migraphx.shaped<1x7x3xf32, 21x3x1>
    return %0 : !migraphx.shaped<1x7x3xf32, 21x3x1>
  }
}
