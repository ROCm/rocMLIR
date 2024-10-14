// RUN: rocmlir-gen -fut mlir_dot_multi_reduce --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_dot_multi_reduce_wrapper --verifier clone -relDiff_threshold 0.01 -RMS_threshold 0.01 -absDiff_threshold 1.2 -| rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2

// We need a check for each output as this test case has three outputs in it.
// CHECK: [1 1 1]
// CHECK: [1 1 1]
module {
  func.func @mlir_dot_multi_reduce(%arg0: !migraphx.shaped<2x32x10x64x64xf16, 0x10x1x20480x320>, %arg1: !migraphx.shaped<2x32x10x64x64xf16, 1310720x40960x4096x64x1>, %arg2: !migraphx.shaped<2x4096x320xf16, 1310720x320x1>, %arg3: !migraphx.shaped<320x320xf16, 320x1>) -> (!migraphx.shaped<2x320x1xf32, 320x1x1>, !migraphx.shaped<2x32x10x64x64xf16, 1310720x40960x4096x64x1>) // attributes {arch = "gfx942:sramecc+:xnack-", kernel = "mixr", num_cu = 304 : i64} 
  {
    %0 = migraphx.literal(dense<2.441410e-05> : tensor<1xf32>) : <1xf32, 0>
    %1 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [2, 320, 320]} : <320x320xf16, 320x1> -> <2x320x320xf16, 0x320x1>
    %2 = migraphx.dot %arg2, %1 : <2x4096x320xf16, 1310720x320x1>, <2x320x320xf16, 0x320x1> -> <2x4096x320xf16, 1310720x320x1>
    %3 = migraphx.reshape %2 {dims = [2, 64, 64, 32, 10]} : <2x4096x320xf16, 1310720x320x1> -> <2x64x64x32x10xf16, 1310720x20480x320x10x1>
    %4 = migraphx.transpose %3 {permutation = [0, 3, 4, 1, 2]} : <2x64x64x32x10xf16, 1310720x20480x320x10x1> -> <2x32x10x64x64xf16, 1310720x10x1x20480x320>
    %5 = migraphx.add %4, %arg0 : <2x32x10x64x64xf16, 1310720x10x1x20480x320>, <2x32x10x64x64xf16, 0x10x1x20480x320> -> <2x32x10x64x64xf16, 1310720x10x1x20480x320>
    %6 = migraphx.add %5, %arg1 : <2x32x10x64x64xf16, 1310720x10x1x20480x320>, <2x32x10x64x64xf16, 1310720x40960x4096x64x1> -> <2x32x10x64x64xf16, 1310720x40960x4096x64x1>
    %7 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [2, 32, 10, 64, 64]} : <1xf32, 0> -> <2x32x10x64x64xf32, 0x0x0x0x0>
    %8 = migraphx.convert %6 {target_type = 2 : i64} : <2x32x10x64x64xf16, 1310720x40960x4096x64x1> to <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %9 = migraphx.mul %8, %7 : <2x32x10x64x64xf32, 1310720x40960x4096x64x1>, <2x32x10x64x64xf32, 0x0x0x0x0> -> <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %10 = migraphx.reshape %9 {dims = [2, 320, 4096]} : <2x32x10x64x64xf32, 1310720x40960x4096x64x1> -> <2x320x4096xf32, 1310720x4096x1>
    %11 = migraphx.reduce_sum %10 {axes = [2]} : <2x320x4096xf32, 1310720x4096x1> -> <2x320x1xf32, 320x1x1>
    return %11, %6 : !migraphx.shaped<2x320x1xf32, 320x1x1>, !migraphx.shaped<2x32x10x64x64xf16, 1310720x40960x4096x64x1>
  }
}