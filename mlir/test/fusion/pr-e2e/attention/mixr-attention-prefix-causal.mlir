// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -RMS_threshold 0.01  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x1xsi32, 1x1>, %arg1: !migraphx.shaped<1x18x4x64xf16, 4608x64x1152x1>, %arg2: !migraphx.shaped<1x2x1x16x64xf16, 2048x1024x1024x64x1>, %arg3: !migraphx.shaped<1x14x16x64xf16, 14336x1024x64x1>) -> !migraphx.shaped<1x4x896xf16, 3584x896x1> attributes {kernel, arch="gfx950"} {
    %0 = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %1 = migraphx.literal(dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xsi32>) : <16xsi32, 1>
    %2 = migraphx.literal(dense<1.250000e-01> : tensor<1xf16>) : <1xf16, 1>
    %3 = migraphx.literal(dense<[[0], [1], [2], [3]]> : tensor<4x1xsi32>) : <4x1xsi32, 1x1>
    %4 = migraphx.broadcast %1 {axis = 1 : i64, out_lens = [4, 16]} : <16xsi32, 1> -> <4x16xsi32, 0x1>
    %5 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [1, 14, 4, 16]} : <1xf16, 1> -> <1x14x4x16xf16, 0x0x0x0>
    %6 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [4, 1]} : <1x1xsi32, 1x1> -> <4x1xsi32, 0x1>
    %7 = migraphx.add %3, %6 : <4x1xsi32, 1x1>, <4x1xsi32, 0x1> -> <4x1xsi32, 1x1>
    %8 = migraphx.multibroadcast %7 {out_dyn_dims = [], out_lens = [4, 16]} : <4x1xsi32, 1x1> -> <4x16xsi32, 1x0>
    %9 = migraphx.greater %4, %8 : <4x16xsi32, 0x1>, <4x16xsi32, 1x0> -> <4x16xsi32, 16x1>
    %10 = migraphx.convert %9 {target_type = 0 : i64} : <4x16xsi32, 16x1> to <4x16xsi8, 16x1>
    %11 = migraphx.multibroadcast %10 {out_dyn_dims = [], out_lens = [1, 14, 4, 16]} : <4x16xsi8, 16x1> -> <1x14x4x16xsi8, 0x0x16x1>
    %12 = migraphx.slice %arg1 {axes = [1], ends = [14], starts = [0]} : <1x18x4x64xf16, 4608x64x1152x1> -> <1x14x4x64xf16, 4608x64x1152x1>
    %13 = migraphx.transpose %arg2 {permutation = [0, 1, 2, 4, 3]} : <1x2x1x16x64xf16, 2048x1024x1024x64x1> -> <1x2x1x64x16xf16, 2048x1024x1024x1x64>
    %14 = migraphx.multibroadcast %13 {out_dyn_dims = [], out_lens = [1, 2, 7, 64, 16]} : <1x2x1x64x16xf16, 2048x1024x1024x1x64> -> <1x2x7x64x16xf16, 2048x1024x0x1x64>
    %15 = migraphx.reshape %14 {dims = [1, 14, 64, 16]} : <1x2x7x64x16xf16, 2048x1024x0x1x64> -> <1x14x64x16xf16, 14336x1024x16x1>
    %16 = migraphx.dot %12, %15 : <1x14x4x64xf16, 4608x64x1152x1>, <1x14x64x16xf16, 14336x1024x16x1> -> <1x14x4x16xf16, 896x64x16x1>
    %17 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 14, 4, 16]} : <1xf16, 1> -> <1x14x4x16xf16, 0x0x0x0>
    %18 = migraphx.mul %16, %5 : <1x14x4x16xf16, 896x64x16x1>, <1x14x4x16xf16, 0x0x0x0> -> <1x14x4x16xf16, 896x64x16x1>
    %19 = migraphx.where %11, %17, %18 : <1x14x4x16xsi8, 0x0x16x1>, <1x14x4x16xf16, 0x0x0x0>, <1x14x4x16xf16, 896x64x16x1> -> <1x14x4x16xf16, 896x64x16x1>
    %20 = migraphx.convert %19 {target_type = 2 : i64} : <1x14x4x16xf16, 896x64x16x1> to <1x14x4x16xf32, 896x64x16x1>
    %21 = migraphx.reshape %20 {dims = [1, 14, 4, 16]} : <1x14x4x16xf32, 896x64x16x1> -> <1x14x4x16xf32, 896x64x16x1>
    %22 = migraphx.reduce_max %21 {axes = [3]} : <1x14x4x16xf32, 896x64x16x1> -> <1x14x4x1xf32, 56x4x1x1>
    %23 = migraphx.reshape %22 {dims = [1, 14, 4, 1]} : <1x14x4x1xf32, 56x4x1x1> -> <1x14x4x1xf32, 56x4x1x1>
    %24 = migraphx.multibroadcast %23 {out_dyn_dims = [], out_lens = [1, 14, 4, 16]} : <1x14x4x1xf32, 56x4x1x1> -> <1x14x4x16xf32, 56x4x1x0>
    %25 = migraphx.sub %20, %24 : <1x14x4x16xf32, 896x64x16x1>, <1x14x4x16xf32, 56x4x1x0> -> <1x14x4x16xf32, 896x64x16x1>
    %26 = migraphx.exp %25 : <1x14x4x16xf32, 896x64x16x1> -> <1x14x4x16xf32, 896x64x16x1>
    %27 = migraphx.reshape %26 {dims = [1, 14, 4, 16]} : <1x14x4x16xf32, 896x64x16x1> -> <1x14x4x16xf32, 896x64x16x1>
    %28 = migraphx.reduce_sum %27 {axes = [3]} : <1x14x4x16xf32, 896x64x16x1> -> <1x14x4x1xf32, 56x4x1x1>
    %29 = migraphx.reshape %28 {dims = [1, 14, 4, 1]} : <1x14x4x1xf32, 56x4x1x1> -> <1x14x4x1xf32, 56x4x1x1>
    %30 = migraphx.multibroadcast %29 {out_dyn_dims = [], out_lens = [1, 14, 4, 16]} : <1x14x4x1xf32, 56x4x1x1> -> <1x14x4x16xf32, 56x4x1x0>
    %31 = migraphx.div %26, %30 : <1x14x4x16xf32, 896x64x16x1>, <1x14x4x16xf32, 56x4x1x0> -> <1x14x4x16xf32, 896x64x16x1>
    %32 = migraphx.convert %31 {target_type = 1 : i64} : <1x14x4x16xf32, 896x64x16x1> to <1x14x4x16xf16, 896x64x16x1>
    %33 = migraphx.dot %32, %arg3 : <1x14x4x16xf16, 896x64x16x1>, <1x14x16x64xf16, 14336x1024x64x1> -> <1x14x4x64xf16, 3584x256x64x1>
    %34 = migraphx.transpose %33 {permutation = [0, 2, 1, 3]} : <1x14x4x64xf16, 3584x256x64x1> -> <1x4x14x64xf16, 3584x64x256x1>
    %35 = migraphx.reshape %34 {dims = [1, 4, 896]} : <1x4x14x64xf16, 3584x64x256x1> -> <1x4x896xf16, 3584x896x1>
    return %35 : !migraphx.shaped<1x4x896xf16, 3584x896x1>
  }
}
