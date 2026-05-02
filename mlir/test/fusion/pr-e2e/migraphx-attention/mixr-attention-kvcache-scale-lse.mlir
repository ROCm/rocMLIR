// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -rand_min_int 32 -rand_max_int 64 -rand_type_int_for_inputs=3 -fut mlir_attention_wrapper -RMS_threshold 0.002 -relDiff_threshold 0.005 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: [1 1 1]
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x2x1x64xf16, 128x64x64x1>,
                                     %arg1: !migraphx.shaped<1x2x64x64xf16, 8192x4096x64x1>,
                                     %arg2: !migraphx.shaped<1x2x64x64xf16, 8192x4096x64x1>,
                                     %arg3: !migraphx.shaped<1x2xi32, 2x1>,
                                     %arg4: !migraphx.shaped<1x2x1x64xf16, 128x64x64x1>)
                                     -> (!migraphx.shaped<1x2x1x64xf16, 128x64x64x1>,
                                         !migraphx.shaped<1x2x1xf32, 2x1x1>) {
    %0, %1 = migraphx.attention %arg0, %arg1, %arg2
      pre_softmax_inputs(%arg4 : !migraphx.shaped<1x2x1x64xf16, 128x64x64x1>)
      current_seq_len(%arg3 : !migraphx.shaped<1x2xi32, 2x1>) {
      ^bb0(%qk: !migraphx.shaped<1x2x1x64xf16, 128x64x64x1>,
           %s: !migraphx.shaped<1x2x1x64xf16, 128x64x64x1>):
        %scaled = migraphx.mul %qk, %s
          : <1x2x1x64xf16, 128x64x64x1>, <1x2x1x64xf16, 128x64x64x1>
          -> <1x2x1x64xf16, 128x64x64x1>
        migraphx.yield %scaled : !migraphx.shaped<1x2x1x64xf16, 128x64x64x1>
      } softmax_type = f32 features = kvcache
      : <1x2x1x64xf16, 128x64x64x1>, <1x2x64x64xf16, 8192x4096x64x1>, <1x2x64x64xf16, 8192x4096x64x1>
      -> <1x2x1x64xf16, 128x64x64x1>, !migraphx.shaped<1x2x1xf32, 2x1x1>
    return %0, %1 : !migraphx.shaped<1x2x1x64xf16, 128x64x64x1>, !migraphx.shaped<1x2x1xf32, 2x1x1>
  }
}
