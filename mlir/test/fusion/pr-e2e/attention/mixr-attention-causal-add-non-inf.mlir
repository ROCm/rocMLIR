// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

module {
  // CHECK: [1 1 1]
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x2x8x64xf16, 1024x512x64x1>, %arg1: !migraphx.shaped<1x2x8x64xf16, 1024x64x128x1>, %arg2: !migraphx.shaped<1x2x8x64xf16, 1024x512x64x1>) -> !migraphx.shaped<1x2x8x64xf16, 1024x512x64x1> attributes {rock.kernel} {
    %0 = migraphx.literal(dense<[[[[-0.000000e+00, 0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF], [-0.000000e+00, -0.000000e+00, 0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF], [-0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF], [-0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFBFF, 0xFBFF, 0xFBFF, 0xFBFF], [-0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFBFF, 0xFBFF, 0xFBFF], [-0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFBFF, 0xFBFF], [-0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFBFF], [-0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00]]]]> : tensor<1x1x8x8xf16>) : <1x1x8x8xf16, 64x64x8x1>
    %1 = migraphx.literal(dense<3.535160e-01> : tensor<1xf16>) : <1xf16, 0>
    %2 = migraphx.reshape %arg0 {dims = [1, 2, 1, 8, 64]} : <1x2x8x64xf16, 1024x512x64x1> -> <1x2x1x8x64xf16, 1024x512x512x64x1>
    %3 = migraphx.transpose %2 {permutation = [0, 1, 2, 4, 3]} : <1x2x1x8x64xf16, 1024x512x512x64x1> -> <1x2x1x64x8xf16, 1024x512x512x1x64>
    %4 = migraphx.multibroadcast %3 {out_dyn_dims = [], out_lens = [1, 2, 1, 64, 8]} : <1x2x1x64x8xf16, 1024x512x512x1x64> -> <1x2x1x64x8xf16, 1024x512x512x1x64>
    %5 = migraphx.reshape %4 {dims = [1, 2, 64, 8]} : <1x2x1x64x8xf16, 1024x512x512x1x64> -> <1x2x64x8xf16, 1024x512x8x1>
    %6 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 2, 64, 8]} : <1xf16, 0> -> <1x2x64x8xf16, 0x0x0x0>
    %7 = migraphx.mul %5, %6 : <1x2x64x8xf16, 1024x512x8x1>, <1x2x64x8xf16, 0x0x0x0> -> <1x2x64x8xf16, 1024x512x8x1>
    %8 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 2, 8, 8]} : <1x1x8x8xf16, 64x64x8x1> -> <1x2x8x8xf16, 64x0x8x1>
    %9 = migraphx.dot %arg1, %7 : <1x2x8x64xf16, 1024x64x128x1>, <1x2x64x8xf16, 1024x512x8x1> -> <1x2x8x8xf16, 128x64x8x1>
    %scores_f32 = migraphx.convert %9 {target_type = 2 : i64} : <1x2x8x8xf16, 128x64x8x1> to <1x2x8x8xf32, 128x64x8x1>
    %mask_f32 = migraphx.convert %8 {target_type = 2 : i64} : <1x2x8x8xf16, 64x0x8x1> to <1x2x8x8xf32, 64x0x8x1>
    %10 = migraphx.add %scores_f32, %mask_f32 : <1x2x8x8xf32, 128x64x8x1>, <1x2x8x8xf32, 64x0x8x1> -> <1x2x8x8xf32, 128x64x8x1>
    %11 = migraphx.reshape %10 {dims = [1, 2, 8, 8]} : <1x2x8x8xf32, 128x64x8x1> -> <1x2x8x8xf32, 128x64x8x1>
    %12 = migraphx.reduce_max %11 {axes = [3]} : <1x2x8x8xf32, 128x64x8x1> -> <1x2x8x1xf32, 16x8x1x1>
    %13 = migraphx.reshape %12 {dims = [1, 2, 8, 1]} : <1x2x8x1xf32, 16x8x1x1> -> <1x2x8x1xf32, 16x8x1x1>
    %14 = migraphx.multibroadcast %13 {out_dyn_dims = [], out_lens = [1, 2, 8, 8]} : <1x2x8x1xf32, 16x8x1x1> -> <1x2x8x8xf32, 16x8x1x0>
    %15 = migraphx.sub %10, %14 : <1x2x8x8xf32, 128x64x8x1>, <1x2x8x8xf32, 16x8x1x0> -> <1x2x8x8xf32, 128x64x8x1>
    %16 = migraphx.exp %15 : <1x2x8x8xf32, 128x64x8x1> -> <1x2x8x8xf32, 128x64x8x1>
    %17 = migraphx.reshape %16 {dims = [1, 2, 8, 8]} : <1x2x8x8xf32, 128x64x8x1> -> <1x2x8x8xf32, 128x64x8x1>
    %18 = migraphx.reduce_sum %17 {axes = [3]} : <1x2x8x8xf32, 128x64x8x1> -> <1x2x8x1xf32, 16x8x1x1>
    %19 = migraphx.reshape %18 {dims = [1, 2, 8, 1]} : <1x2x8x1xf32, 16x8x1x1> -> <1x2x8x1xf32, 16x8x1x1>
    %20 = migraphx.multibroadcast %19 {out_dyn_dims = [], out_lens = [1, 2, 8, 8]} : <1x2x8x1xf32, 16x8x1x1> -> <1x2x8x8xf32, 16x8x1x0>
    %21 = migraphx.div %16, %20 : <1x2x8x8xf32, 128x64x8x1>, <1x2x8x8xf32, 16x8x1x0> -> <1x2x8x8xf32, 128x64x8x1>
    %22 = migraphx.convert %21 {target_type = 1 : i64} : <1x2x8x8xf32, 128x64x8x1> to <1x2x8x8xf16, 128x64x8x1>
    %23 = migraphx.dot %22, %arg2 : <1x2x8x8xf16, 128x64x8x1>, <1x2x8x64xf16, 1024x512x64x1> -> <1x2x8x64xf16, 1024x512x64x1>
    return %23 : !migraphx.shaped<1x2x8x64xf16, 1024x512x64x1>
  }
}
