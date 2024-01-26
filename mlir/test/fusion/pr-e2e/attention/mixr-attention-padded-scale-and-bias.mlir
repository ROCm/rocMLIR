// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.000004  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x7x3xf32, 21x3x1> {func.read_access}, 
                                    %arg1: !migraphx.shaped<1x3x7xf32, 21x7x1> {func.read_access}, 
                                    %arg2: !migraphx.shaped<1x7x3xf32, 21x3x1> {func.read_access}, 
                                    %arg3: !migraphx.shaped<1x7x7xf32, 49x7x1> {func.read_access},
                                    %arg4: !migraphx.shaped<1x7x7xf32, 49x7x1> {func.read_access})
                                    -> (!migraphx.shaped<1x7x3xf32, 21x3x1> {func.write_access}) {
    %0 = migraphx.dot %arg0, %arg1: <1x7x3xf32, 21x3x1>, <1x3x7xf32, 21x7x1> -> <1x7x7xf32, 49x7x1>
    %scaled = migraphx.mul %0, %arg3 : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1> -> <1x7x7xf32, 49x7x1>
    %biased = migraphx.add %scaled, %arg4 : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1> -> <1x7x7xf32, 49x7x1>
    %1 = migraphx.softmax %biased{axis = 2 : i64} : <1x7x7xf32, 49x7x1> -> <1x7x7xf32, 49x7x1>
    %2 = migraphx.dot %1, %arg2: <1x7x7xf32, 49x7x1>, <1x7x3xf32, 21x3x1> -> <1x7x3xf32, 21x3x1>
    return %2 : !migraphx.shaped<1x7x3xf32, 21x3x1>
  }
}
