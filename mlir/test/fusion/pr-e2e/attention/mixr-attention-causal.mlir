// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x96x2x128xf16, 24576x128x12288x1>, %arg1: !migraphx.shaped<1x32x64x128xf16, 262144x8192x128x1>, %arg2: !migraphx.shaped<1x32x64x128xf16, 262144x8192x128x1>) -> !migraphx.shaped<1x2x4096xf16, 8192x4096x1> {
    %0 = migraphx.literal(dense<[0, 1]> : tensor<2xsi32>) : <2xsi32, 1>
    %1 = migraphx.literal(dense<8.837890e-02> : tensor<1xf16>) : <1xf16, 1>
    %2 = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %3 = migraphx.literal(dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xsi32>) : <64xsi32, 1>
    %4 = migraphx.slice %arg0 {axes = [1], ends = [32], starts = [0]} : <1x96x2x128xf16, 24576x128x12288x1> -> <1x32x2x128xf16, 24576x128x12288x1>
    %5 = migraphx.transpose %arg1 {permutation = [0, 1, 3, 2]} : <1x32x64x128xf16, 262144x8192x128x1> -> <1x32x128x64xf16, 262144x8192x1x128>
    %6 = migraphx.dot %4, %5 : <1x32x2x128xf16, 24576x128x12288x1>, <1x32x128x64xf16, 262144x8192x1x128> -> <1x32x2x64xf16, 4096x128x64x1>
    %7 = migraphx.multibroadcast %3 {out_dyn_dims = [], out_lens = [1, 32, 2, 64]} : <64xsi32, 1> -> <1x32x2x64xsi32, 0x0x0x1>
    %8 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [1, 32, 2, 64]} : <1xf16, 1> -> <1x32x2x64xf16, 0x0x0x0>
    %9 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 32, 2, 64]} : <1xf16, 1> -> <1x32x2x64xf16, 0x0x0x0>
    %fused = migraphx.mul %6, %9 : <1x32x2x64xf16, 4096x128x64x1>, <1x32x2x64xf16, 0x0x0x0> -> <1x32x2x64xf16, 4096x128x64x1>
    %10 = migraphx.reshape %0 {dims = [2, 1]} : <2xsi32, 1> -> <2x1xsi32, 1x1>
    %11 = migraphx.multibroadcast %10 {out_dyn_dims = [], out_lens = [1, 32, 2, 64]} : <2x1xsi32, 1x1> -> <1x32x2x64xsi32, 0x0x1x0>
    %12 = migraphx.greater %7, %11 : <1x32x2x64xsi32, 0x0x0x1>, <1x32x2x64xsi32, 0x0x1x0> -> <1x32x2x64xsi32, 4096x128x64x1>
    %13 = migraphx.convert %12 {target_type = 0 : i64} : <1x32x2x64xsi32, 4096x128x64x1> to <1x32x2x64xsi8, 4096x128x64x1>
    %14 = migraphx.where %13, %8, %fused : <1x32x2x64xsi8, 4096x128x64x1>, <1x32x2x64xf16, 0x0x0x0>, <1x32x2x64xf16, 4096x128x64x1> -> <1x32x2x64xf16, 4096x128x64x1>
    %21 = migraphx.softmax %14 {axis = 3 : i64} : <1x32x2x64xf16, 4096x128x64x1> -> <1x32x2x64xf16, 4096x128x64x1>
    %22 = migraphx.dot %21, %arg2 : <1x32x2x64xf16, 4096x128x64x1>, <1x32x64x128xf16, 262144x8192x128x1> -> <1x32x2x128xf16, 8192x256x128x1>
    %23 = migraphx.transpose %22 {permutation = [0, 2, 1, 3]} : <1x32x2x128xf16, 8192x256x128x1> -> <1x2x32x128xf16, 8192x128x256x1>
    %24 = migraphx.reshape %23 {dims = [1, 2, 4096]} : <1x2x32x128xf16, 8192x128x256x1> -> <1x2x4096xf16, 8192x4096x1>
    return %24 : !migraphx.shaped<1x2x4096xf16, 8192x4096x1>
  }
}
