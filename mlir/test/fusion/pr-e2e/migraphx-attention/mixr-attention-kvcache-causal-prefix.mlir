// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand_min_int 32 -rand_max_int 64 -rand_type_int_for_inputs=3 -rand_min_int 0 -rand_max_int 16 -rand_type_int_for_inputs=4 -rand 1 -rand_type float -fut mlir_attention_wrapper -RMS_threshold 0.03 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x64x64xf16, 4096x64x1>,
                                     %arg1: !migraphx.shaped<1x64x64xf16, 4096x64x1>,
                                     %arg2: !migraphx.shaped<1x64x64xf16, 4096x64x1>,
                                     %arg3: !migraphx.shaped<1xi32, 1>,
                                     %arg4: !migraphx.shaped<1xi32, 1>)
                                     -> !migraphx.shaped<1x64x64xf16, 4096x64x1> {
    %0 = migraphx.attention %arg0, %arg1, %arg2
      current_seq_len(%arg3 : !migraphx.shaped<1xi32, 1>)
      prefix_offset(%arg4 : !migraphx.shaped<1xi32, 1>) {
    } features = "kvcache|causal|prefix_offset"
      : <1x64x64xf16, 4096x64x1>, <1x64x64xf16, 4096x64x1>, <1x64x64xf16, 4096x64x1>
      -> <1x64x64xf16, 4096x64x1>
    return %0 : !migraphx.shaped<1x64x64xf16, 4096x64x1>
  }
}
