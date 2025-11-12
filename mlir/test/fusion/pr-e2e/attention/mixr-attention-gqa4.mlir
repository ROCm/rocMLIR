// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.000004  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<2x18x4x64xf16, 4608x64x1152x1>, %arg1: !migraphx.shaped<2x2x8x64xf16, 1024x512x64x1>, %arg2: !migraphx.shaped<2x2x8x64xf16, 1024x512x64x1>, %arg3: !migraphx.shaped<2x1xsi32, 1x1>) -> !migraphx.shaped<2x4x896xf16, 3584x896x1> {
    %0 = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %1 = migraphx.literal(dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xsi32>) : <8xsi32, 1>
    %2 = migraphx.literal(dense<1.250000e-01> : tensor<1xf16>) : <1xf16, 1>
    %3 = migraphx.literal(dense<[[0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]]> : tensor<4x8xsi8>) : <4x8xsi8, 8x1>
    %4 = migraphx.slice %arg0 {axes = [1], ends = [14], starts = [0]} : <2x18x4x64xf16, 4608x64x1152x1> -> <2x14x4x64xf16, 4608x64x1152x1>
    %5 = migraphx.reshape %arg1 {dims = [2, 2, 1, 8, 64]} : <2x2x8x64xf16, 1024x512x64x1> -> <2x2x1x8x64xf16, 1024x512x512x64x1>
    %6 = migraphx.multibroadcast %5 {out_dyn_dims = [], out_lens = [2, 2, 7, 8, 64]} : <2x2x1x8x64xf16, 1024x512x512x64x1> -> <2x2x7x8x64xf16, 1024x512x0x64x1>
    %7 = migraphx.reshape %6 {dims = [2, 14, 8, 64]} : <2x2x7x8x64xf16, 1024x512x0x64x1> -> <2x14x8x64xf16, 7168x512x64x1>
    %8 = migraphx.reshape %arg2 {dims = [2, 2, 1, 8, 64]} : <2x2x8x64xf16, 1024x512x64x1> -> <2x2x1x8x64xf16, 1024x512x512x64x1>
    %9 = migraphx.transpose %8 {permutation = [0, 1, 2, 4, 3]} : <2x2x1x8x64xf16, 1024x512x512x64x1> -> <2x2x1x64x8xf16, 1024x512x512x1x64>
    %10 = migraphx.multibroadcast %9 {out_dyn_dims = [], out_lens = [2, 2, 7, 64, 8]} : <2x2x1x64x8xf16, 1024x512x512x1x64> -> <2x2x7x64x8xf16, 1024x512x0x1x64>
    %11 = migraphx.reshape %10 {dims = [2, 14, 64, 8]} : <2x2x7x64x8xf16, 1024x512x0x1x64> -> <2x14x64x8xf16, 7168x512x8x1>
    %12 = migraphx.dot %4, %11 : <2x14x4x64xf16, 4608x64x1152x1>, <2x14x64x8xf16, 7168x512x8x1> -> <2x14x4x8xf16, 448x32x8x1>
    %13 = migraphx.broadcast %1 {axis = 1 : i64, out_lens = [2, 8]} : <8xsi32, 1> -> <2x8xsi32, 0x1>
    %14 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [2, 14, 4, 8]} : <1xf16, 1> -> <2x14x4x8xf16, 0x0x0x0>
    %15 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [2, 14, 4, 8]} : <1xf16, 1> -> <2x14x4x8xf16, 0x0x0x0>
    %16 = migraphx.mul %12, %15 : <2x14x4x8xf16, 448x32x8x1>, <2x14x4x8xf16, 0x0x0x0> -> <2x14x4x8xf16, 448x32x8x1>
    %17 = migraphx.multibroadcast %3 {out_dyn_dims = [], out_lens = [2, 14, 4, 8]} : <4x8xsi8, 8x1> -> <2x14x4x8xsi8, 0x0x8x1>
    %18 = migraphx.where %17, %14, %16 : <2x14x4x8xsi8, 0x0x8x1>, <2x14x4x8xf16, 0x0x0x0>, <2x14x4x8xf16, 448x32x8x1> -> <2x14x4x8xf16, 448x32x8x1>
    %19 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [2, 8]} : <2x1xsi32, 1x1> -> <2x8xsi32, 1x0>
    %20 = migraphx.greater %13, %19 : <2x8xsi32, 0x1>, <2x8xsi32, 1x0> -> <2x8xsi32, 8x1>
    %21 = migraphx.convert %20 {target_type = 0 : i64} : <2x8xsi32, 8x1> to <2x8xsi8, 8x1>
    %22 = migraphx.reshape %21 {dims = [2, 1, 1, 8]} : <2x8xsi8, 8x1> -> <2x1x1x8xsi8, 8x8x8x1>
    %23 = migraphx.multibroadcast %22 {out_dyn_dims = [], out_lens = [2, 14, 4, 8]} : <2x1x1x8xsi8, 8x8x8x1> -> <2x14x4x8xsi8, 8x0x0x1>
    %24 = migraphx.where %23, %14, %18 : <2x14x4x8xsi8, 8x0x0x1>, <2x14x4x8xf16, 0x0x0x0>, <2x14x4x8xf16, 448x32x8x1> -> <2x14x4x8xf16, 448x32x8x1>
    %25 = migraphx.convert %24 {target_type = 2 : i64} : <2x14x4x8xf16, 448x32x8x1> to <2x14x4x8xf32, 448x32x8x1>
    %26 = migraphx.reshape %25 {dims = [2, 14, 4, 8]} : <2x14x4x8xf32, 448x32x8x1> -> <2x14x4x8xf32, 448x32x8x1>
    %27 = migraphx.reduce_max %26 {axes = [3]} : <2x14x4x8xf32, 448x32x8x1> -> <2x14x4x1xf32, 56x4x1x1>
    %28 = migraphx.reshape %27 {dims = [2, 14, 4, 1]} : <2x14x4x1xf32, 56x4x1x1> -> <2x14x4x1xf32, 56x4x1x1>
    %29 = migraphx.multibroadcast %28 {out_dyn_dims = [], out_lens = [2, 14, 4, 8]} : <2x14x4x1xf32, 56x4x1x1> -> <2x14x4x8xf32, 56x4x1x0>
    %30 = migraphx.sub %25, %29 : <2x14x4x8xf32, 448x32x8x1>, <2x14x4x8xf32, 56x4x1x0> -> <2x14x4x8xf32, 448x32x8x1>
    %31 = migraphx.exp %30 : <2x14x4x8xf32, 448x32x8x1> -> <2x14x4x8xf32, 448x32x8x1>
    %32 = migraphx.reshape %31 {dims = [2, 14, 4, 8]} : <2x14x4x8xf32, 448x32x8x1> -> <2x14x4x8xf32, 448x32x8x1>
    %33 = migraphx.reduce_sum %32 {axes = [3]} : <2x14x4x8xf32, 448x32x8x1> -> <2x14x4x1xf32, 56x4x1x1>
    %34 = migraphx.reshape %33 {dims = [2, 14, 4, 1]} : <2x14x4x1xf32, 56x4x1x1> -> <2x14x4x1xf32, 56x4x1x1>
    %35 = migraphx.multibroadcast %34 {out_dyn_dims = [], out_lens = [2, 14, 4, 8]} : <2x14x4x1xf32, 56x4x1x1> -> <2x14x4x8xf32, 56x4x1x0>
    %36 = migraphx.div %31, %35 : <2x14x4x8xf32, 448x32x8x1>, <2x14x4x8xf32, 56x4x1x0> -> <2x14x4x8xf32, 448x32x8x1>
    %37 = migraphx.convert %36 {target_type = 1 : i64} : <2x14x4x8xf32, 448x32x8x1> to <2x14x4x8xf16, 448x32x8x1>
    %38 = migraphx.dot %37, %7 : <2x14x4x8xf16, 448x32x8x1>, <2x14x8x64xf16, 7168x512x64x1> -> <2x14x4x64xf16, 3584x256x64x1>
    %39 = migraphx.transpose %38 {permutation = [0, 2, 1, 3]} : <2x14x4x64xf16, 3584x256x64x1> -> <2x4x14x64xf16, 3584x64x256x1>
    %40 = migraphx.reshape %39 {dims = [2, 4, 896]} : <2x4x14x64xf16, 3584x64x256x1> -> <2x4x896xf16, 3584x896x1>
    return %40 : !migraphx.shaped<2x4x896xf16, 3584x896x1>
  }
}
