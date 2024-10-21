// RUN: rocmlir-gen -fut mlir_convolution_multi_reduce --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_convolution_multi_reduce_wrapper --verifier clone -relDiff_threshold 0.01 -RMS_threshold 0.01 -absDiff_threshold 0.4 -| rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2

// We need a check for each output as this test case has three outputs in it.
// CHECK: [1 1 1]
// CHECK: [1 1 1]
// CHECK: [1 1 1]
module {
  func.func @mlir_convolution_multi_reduce(%arg0: !migraphx.shaped<2x32x10x64x64xf32, 0x10x1x0x0>, %arg1: !migraphx.shaped<2x4x64x64xf32, 16384x4096x64x1>, %arg2: !migraphx.shaped<320x4x3x3xf32, 36x9x3x1>) -> (!migraphx.shaped<2x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<2x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<2x32x10x64x64xf32, 1310720x40960x4096x64x1>) // attributes {arch = "gfx942:sramecc+:xnack-", kernel = "mixr"} 
  {
    %0 = migraphx.literal(dense<2.44140629E-5> : tensor<1xf32>) : <1xf32, 0>
    %1 = migraphx.literal(dense<2.44140629E-5> : tensor<1xf32>) : <1xf32, 0>
    %2 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <2x4x64x64xf32, 16384x4096x64x1>, <320x4x3x3xf32, 36x9x3x1> -> <2x320x64x64xf32, 1310720x4096x64x1>
    %3 = migraphx.reshape %2 {dims = [2, 32, 10, 64, 64]} : <2x320x64x64xf32, 1310720x4096x64x1> -> <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %4 = migraphx.add %3, %arg0 : <2x32x10x64x64xf32, 1310720x40960x4096x64x1>, <2x32x10x64x64xf32, 0x10x1x0x0> -> <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %5 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [2, 32, 10, 64, 64]} : <1xf32, 0> -> <2x32x10x64x64xf32, 0x0x0x0x0>
    %6 = migraphx.mul %4, %5 : <2x32x10x64x64xf32, 1310720x40960x4096x64x1>, <2x32x10x64x64xf32, 0x0x0x0x0> -> <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %7 = migraphx.reshape %6 {dims = [2, 32, 40960]} : <2x32x10x64x64xf32, 1310720x40960x4096x64x1> -> <2x32x40960xf32, 1310720x40960x1>
    %8 = migraphx.reduce_sum %7 {axes = [2]} : <2x32x40960xf32, 1310720x40960x1> -> <2x32x1xf32, 32x1x1>
    %9 = migraphx.reshape %8 {dims = [2, 32, 1, 1, 1]} : <2x32x1xf32, 32x1x1> -> <2x32x1x1x1xf32, 32x1x1x1x1>
    %10 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [2, 32, 10, 64, 64]} : <1xf32, 0> -> <2x32x10x64x64xf32, 0x0x0x0x0>
    %11 = migraphx.mul %4, %4 : <2x32x10x64x64xf32, 1310720x40960x4096x64x1>, <2x32x10x64x64xf32, 1310720x40960x4096x64x1> -> <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %12 = migraphx.mul %11, %10 : <2x32x10x64x64xf32, 1310720x40960x4096x64x1>, <2x32x10x64x64xf32, 0x0x0x0x0> -> <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %13 = migraphx.reshape %12 {dims = [2, 32, 40960]} : <2x32x10x64x64xf32, 1310720x40960x4096x64x1> -> <2x32x40960xf32, 1310720x40960x1>
    %14 = migraphx.reduce_sum %13 {axes = [2]} : <2x32x40960xf32, 1310720x40960x1> -> <2x32x1xf32, 32x1x1>
    %15 = migraphx.reshape %14 {dims = [2, 32, 1, 1, 1]} : <2x32x1xf32, 32x1x1> -> <2x32x1x1x1xf32, 32x1x1x1x1>
    return %9, %15, %4 : !migraphx.shaped<2x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<2x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<2x32x10x64x64xf32, 1310720x40960x4096x64x1>
  }
}
