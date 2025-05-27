// RUN: rocmlir-gen -fut mlir_conv_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_conv_gemm_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
  func.func private @mlir_conv_gemm(%arg0: !migraphx.shaped<2x64x32x32xf32, 65536x1x2048x64>,
                                    %arg1: !migraphx.shaped<64x64x1x1xf32, 64x1x64x64>,
                                    %arg2: !migraphx.shaped<1x64x16xf32, 1024x1x64>,
                                    %arg3: !migraphx.shaped<1x2048x64xf32, 0x0x1>) -> (!migraphx.shaped<1x2048x16xf32, 32768x16x1>) {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <2x64x32x32xf32, 65536x1x2048x64>, <64x64x1x1xf32, 64x1x64x64> -> <2x64x32x32xf32, 65536x1x2048x64>
    %1 = migraphx.transpose %0 {permutation = [0, 2, 3, 1]} : <2x64x32x32xf32, 65536x1x2048x64> -> <2x32x32x64xf32, 65536x2048x64x1>
    %2 = migraphx.reshape %1 {dims = [1, 2048, 64]} : <2x32x32x64xf32, 65536x2048x64x1> -> <1x2048x64xf32, 131072x64x1>
    %3 = migraphx.dot %2, %arg2: <1x2048x64xf32, 131072x64x1>, <1x64x16xf32, 1024x1x64> -> <1x2048x16xf32, 32768x16x1>
    return %3 : !migraphx.shaped<1x2048x16xf32, 32768x16x1>
  }
}
