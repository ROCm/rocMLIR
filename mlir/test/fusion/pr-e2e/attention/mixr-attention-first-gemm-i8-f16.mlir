// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_attention_wrapper -relDiff_threshold 0.000002 -RMS_threshold 0.015  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]

module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x64x32xi8, 2048x32x1> {mhal.read_access},
                                    %arg1: !migraphx.shaped<1x32x64xi8, 2048x64x1> {mhal.read_access},
                                    %arg2: !migraphx.shaped<1x64x32xf16, 2048x32x1> {mhal.read_access},
                                    %arg3: !migraphx.shaped<1x64x64xf16, 4096x64x1> {mhal.read_access},
                                    %qscale: !migraphx.shaped<1x1x1xf16, 1x1x1> {mhal.read_access}) 
                                    -> (!migraphx.shaped<1x64x32xf16, 2048x32x1> {mhal.write_access})
                                    {
    %0 = migraphx.quant_dot %arg0, %arg1: <1x64x32xi8, 2048x32x1>, <1x32x64xi8, 2048x64x1> -> <1x64x64xi32, 4096x64x1>
    %1 = migraphx.dequantizelinear %0, %qscale : <1x64x64xi32, 4096x64x1>, <1x1x1xf16, 1x1x1> -> <1x64x64xf16, 4096x64x1>
    %biased = migraphx.add %1, %arg3 : <1x64x64xf16, 4096x64x1>, <1x64x64xf16, 4096x64x1> -> <1x64x64xf16, 4096x64x1>
    %2 = migraphx.softmax %biased{axis = 2 : i64} : <1x64x64xf16, 4096x64x1> -> <1x64x64xf16, 4096x64x1>
    %3 = migraphx.dot %2, %arg2: <1x64x64xf16, 4096x64x1>, <1x64x32xf16, 2048x32x1> -> <1x64x32xf16, 2048x32x1>
    return %3 : !migraphx.shaped<1x64x32xf16, 2048x32x1>
  }
}
