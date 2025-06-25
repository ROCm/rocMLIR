// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.000004  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: [1 1 1]

module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<2x1x5x64xf16, 320x320x64x1> {mhal.read_access}, %arg1: !migraphx.shaped<2x1x5x64xf16, 320x320x64x1> {mhal.read_access}, %arg2: !migraphx.shaped<2x1x5x64xf16, 320x320x64x1> {mhal.read_access}, %arg3: !migraphx.shaped<5x5xf16, 5x1> {mhal.read_access}) -> (!migraphx.shaped<2x1x5x64xf16, 320x320x64x1> {mhal.write_access}, !migraphx.shaped<2x1x5xf16, 5x5x1> {mhal.write_access}) {
    %0 = migraphx.literal(dense<1.441410e+00> : tensor<1xf16>) : <1xf16, 1>
    %1 = migraphx.literal(dense<1.250000e-01> : tensor<1xf16>) : <1xf16, 1>
    %2 = migraphx.transpose %arg1 {permutation = [0, 1, 3, 2]} : <2x1x5x64xf16, 320x320x64x1> -> <2x1x64x5xf16, 320x320x1x64>
    %3 = migraphx.dot %arg0, %2 : <2x1x5x64xf16, 320x320x64x1>, <2x1x64x5xf16, 320x320x1x64> -> <2x1x5x5xf16, 25x25x5x1>
    %4 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [2, 1, 5, 5]} : <1xf16, 1> -> <2x1x5x5xf16, 0x0x0x0>
    %5 = migraphx.mul %3, %4 : <2x1x5x5xf16, 25x25x5x1>, <2x1x5x5xf16, 0x0x0x0> -> <2x1x5x5xf16, 25x25x5x1>
    %6 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [2, 1, 5, 5]} : <5x5xf16, 5x1> -> <2x1x5x5xf16, 0x0x5x1>
    %7 = migraphx.add %5, %6 : <2x1x5x5xf16, 25x25x5x1>, <2x1x5x5xf16, 0x0x5x1> -> <2x1x5x5xf16, 25x25x5x1>
    %8 = migraphx.reshape %7 {dims = [2, 1, 5, 5]} : <2x1x5x5xf16, 25x25x5x1> -> <2x1x5x5xf16, 25x25x5x1>
    %9 = migraphx.reduce_max %8 {axes = [3]} : <2x1x5x5xf16, 25x25x5x1> -> <2x1x5x1xf16, 5x5x1x1>
    %10 = migraphx.reshape %9 {dims = [2, 1, 5, 1]} : <2x1x5x1xf16, 5x5x1x1> -> <2x1x5x1xf16, 5x5x1x1>
    %11 = migraphx.multibroadcast %10 {out_dyn_dims = [], out_lens = [2, 1, 5, 5]} : <2x1x5x1xf16, 5x5x1x1> -> <2x1x5x5xf16, 5x5x1x0>
    %12 = migraphx.sub %7, %11 : <2x1x5x5xf16, 25x25x5x1>, <2x1x5x5xf16, 5x5x1x0> -> <2x1x5x5xf16, 25x25x5x1>
    %13 = migraphx.exp %12 : <2x1x5x5xf16, 25x25x5x1> -> <2x1x5x5xf16, 25x25x5x1>
    %14 = migraphx.reshape %13 {dims = [2, 1, 5, 5]} : <2x1x5x5xf16, 25x25x5x1> -> <2x1x5x5xf16, 25x25x5x1>
    %15 = migraphx.reduce_sum %14 {axes = [3]} : <2x1x5x5xf16, 25x25x5x1> -> <2x1x5x1xf16, 5x5x1x1>
    %16 = migraphx.reshape %15 {dims = [2, 1, 5, 1]} : <2x1x5x1xf16, 5x5x1x1> -> <2x1x5x1xf16, 5x5x1x1>
    %17 = migraphx.log %16 : <2x1x5x1xf16, 5x5x1x1> -> <2x1x5x1xf16, 5x5x1x1>
    %18 = migraphx.recip %16 : <2x1x5x1xf16, 5x5x1x1> -> <2x1x5x1xf16, 5x5x1x1>
    %19 = migraphx.multibroadcast %18 {out_dyn_dims = [], out_lens = [2, 1, 5, 5]} : <2x1x5x1xf16, 5x5x1x1> -> <2x1x5x5xf16, 5x5x1x0>
    %20 = migraphx.mul %13, %19 : <2x1x5x5xf16, 25x25x5x1>, <2x1x5x5xf16, 5x5x1x0> -> <2x1x5x5xf16, 25x25x5x1>
    %21 = migraphx.add %17, %10 : <2x1x5x1xf16, 5x5x1x1>, <2x1x5x1xf16, 5x5x1x1> -> <2x1x5x1xf16, 5x5x1x1>
    %22 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [2, 1, 5, 1]} : <1xf16, 1> -> <2x1x5x1xf16, 0x0x0x1>
    %23 = migraphx.mul %21, %22 : <2x1x5x1xf16, 5x5x1x1>, <2x1x5x1xf16, 0x0x0x1> -> <2x1x5x1xf16, 5x5x1x1>
    %24 = migraphx.reshape %23 {dims = [2, 1, 5]} : <2x1x5x1xf16, 5x5x1x1> -> <2x1x5xf16, 5x5x1>
    %26 = migraphx.dot %20, %arg2 : <2x1x5x5xf16, 25x25x5x1>, <2x1x5x64xf16, 320x320x64x1> -> <2x1x5x64xf16, 320x320x64x1>
    return %26, %24 : !migraphx.shaped<2x1x5x64xf16, 320x320x64x1>, !migraphx.shaped<2x1x5xf16, 5x5x1>
  }
}
