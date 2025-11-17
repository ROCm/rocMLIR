// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.000004  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<2x18x1x64xf16, 1152x64x64x1>, %arg1: !migraphx.shaped<2x2x8x64xf16, 1024x512x64x1>, %arg2: !migraphx.shaped<2x2x8x64xf16, 1024x512x64x1>, %arg3: !migraphx.shaped<2x1xsi32, 1x1>) -> !migraphx.shaped<2x1x896xf16, 896x896x1> {
    %0 = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %1 = migraphx.literal(dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xsi32>) : <8xsi32, 1>
    %2 = migraphx.literal(dense<1.250000e-01> : tensor<1xf16>) : <1xf16, 1>
    %3 = migraphx.slice %arg0 {axes = [1], ends = [14], starts = [0]} : <2x18x1x64xf16, 1152x64x64x1> -> <2x14x1x64xf16, 1152x64x64x1>
    %4 = migraphx.reshape %arg1 {dims = [2, 2, 1, 8, 64]} : <2x2x8x64xf16, 1024x512x64x1> -> <2x2x1x8x64xf16, 1024x512x512x64x1>
    %5 = migraphx.multibroadcast %4 {out_dyn_dims = [], out_lens = [2, 2, 7, 8, 64]} : <2x2x1x8x64xf16, 1024x512x512x64x1> -> <2x2x7x8x64xf16, 1024x512x0x64x1>
    %6 = migraphx.reshape %5 {dims = [2, 14, 8, 64]} : <2x2x7x8x64xf16, 1024x512x0x64x1> -> <2x14x8x64xf16, 7168x512x64x1>
    %7 = migraphx.reshape %arg2 {dims = [2, 2, 1, 8, 64]} : <2x2x8x64xf16, 1024x512x64x1> -> <2x2x1x8x64xf16, 1024x512x512x64x1>
    %8 = migraphx.transpose %7 {permutation = [0, 1, 2, 4, 3]} : <2x2x1x8x64xf16, 1024x512x512x64x1> -> <2x2x1x64x8xf16, 1024x512x512x1x64>
    %9 = migraphx.multibroadcast %8 {out_dyn_dims = [], out_lens = [2, 2, 7, 64, 8]} : <2x2x1x64x8xf16, 1024x512x512x1x64> -> <2x2x7x64x8xf16, 1024x512x0x1x64>
    %10 = migraphx.reshape %9 {dims = [2, 14, 64, 8]} : <2x2x7x64x8xf16, 1024x512x0x1x64> -> <2x14x64x8xf16, 7168x512x8x1>
    %11 = migraphx.dot %3, %10 : <2x14x1x64xf16, 1152x64x64x1>, <2x14x64x8xf16, 7168x512x8x1> -> <2x14x1x8xf16, 112x8x8x1>
    %12 = migraphx.broadcast %1 {axis = 1 : i64, out_lens = [2, 8]} : <8xsi32, 1> -> <2x8xsi32, 0x1>
    %13 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [2, 14, 1, 8]} : <1xf16, 1> -> <2x14x1x8xf16, 0x0x0x0>
    %14 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [2, 14, 1, 8]} : <1xf16, 1> -> <2x14x1x8xf16, 0x0x0x0>
    %15 = migraphx.mul %11, %14 : <2x14x1x8xf16, 112x8x8x1>, <2x14x1x8xf16, 0x0x0x0> -> <2x14x1x8xf16, 112x8x8x1>
    %16 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [2, 8]} : <2x1xsi32, 1x1> -> <2x8xsi32, 1x0>
    %17 = migraphx.greater %12, %16 : <2x8xsi32, 0x1>, <2x8xsi32, 1x0> -> <2x8xsi32, 8x1>
    %18 = migraphx.convert %17 {target_type = 0 : i64} : <2x8xsi32, 8x1> to <2x8xsi8, 8x1>
    %19 = migraphx.reshape %18 {dims = [2, 1, 1, 8]} : <2x8xsi8, 8x1> -> <2x1x1x8xsi8, 8x8x8x1>
    %20 = migraphx.multibroadcast %19 {out_dyn_dims = [], out_lens = [2, 14, 1, 8]} : <2x1x1x8xsi8, 8x8x8x1> -> <2x14x1x8xsi8, 8x0x8x1>
    %21 = migraphx.where %20, %13, %15 : <2x14x1x8xsi8, 8x0x8x1>, <2x14x1x8xf16, 0x0x0x0>, <2x14x1x8xf16, 112x8x8x1> -> <2x14x1x8xf16, 112x8x8x1>
    %22 = migraphx.convert %21 {target_type = 2 : i64} : <2x14x1x8xf16, 112x8x8x1> to <2x14x1x8xf32, 112x8x8x1>
    %23 = migraphx.reshape %22 {dims = [2, 14, 1, 8]} : <2x14x1x8xf32, 112x8x8x1> -> <2x14x1x8xf32, 112x8x8x1>
    %24 = migraphx.reduce_max %23 {axes = [3]} : <2x14x1x8xf32, 112x8x8x1> -> <2x14x1x1xf32, 14x1x1x1>
    %25 = migraphx.reshape %24 {dims = [2, 14, 1, 1]} : <2x14x1x1xf32, 14x1x1x1> -> <2x14x1x1xf32, 14x1x1x1>
    %26 = migraphx.multibroadcast %25 {out_dyn_dims = [], out_lens = [2, 14, 1, 8]} : <2x14x1x1xf32, 14x1x1x1> -> <2x14x1x8xf32, 14x1x1x0>
    %27 = migraphx.sub %22, %26 : <2x14x1x8xf32, 112x8x8x1>, <2x14x1x8xf32, 14x1x1x0> -> <2x14x1x8xf32, 112x8x8x1>
    %28 = migraphx.exp %27 : <2x14x1x8xf32, 112x8x8x1> -> <2x14x1x8xf32, 112x8x8x1>
    %29 = migraphx.reshape %28 {dims = [2, 14, 1, 8]} : <2x14x1x8xf32, 112x8x8x1> -> <2x14x1x8xf32, 112x8x8x1>
    %30 = migraphx.reduce_sum %29 {axes = [3]} : <2x14x1x8xf32, 112x8x8x1> -> <2x14x1x1xf32, 14x1x1x1>
    %31 = migraphx.reshape %30 {dims = [2, 14, 1, 1]} : <2x14x1x1xf32, 14x1x1x1> -> <2x14x1x1xf32, 14x1x1x1>
    %32 = migraphx.multibroadcast %31 {out_dyn_dims = [], out_lens = [2, 14, 1, 8]} : <2x14x1x1xf32, 14x1x1x1> -> <2x14x1x8xf32, 14x1x1x0>
    %33 = migraphx.div %28, %32 : <2x14x1x8xf32, 112x8x8x1>, <2x14x1x8xf32, 14x1x1x0> -> <2x14x1x8xf32, 112x8x8x1>
    %34 = migraphx.convert %33 {target_type = 1 : i64} : <2x14x1x8xf32, 112x8x8x1> to <2x14x1x8xf16, 112x8x8x1>
    %35 = migraphx.dot %34, %6 : <2x14x1x8xf16, 112x8x8x1>, <2x14x8x64xf16, 7168x512x64x1> -> <2x14x1x64xf16, 896x64x64x1>
    %36 = migraphx.transpose %35 {permutation = [0, 2, 1, 3]} : <2x14x1x64xf16, 896x64x64x1> -> <2x1x14x64xf16, 896x64x64x1>
    %37 = migraphx.reshape %36 {dims = [2, 1, 896]} : <2x1x14x64xf16, 896x64x64x1> -> <2x1x896xf16, 896x896x1>
    return %37 : !migraphx.shaped<2x1x896xf16, 896x896x1>
  }
}
