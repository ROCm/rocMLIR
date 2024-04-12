// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_reshape_convolution_real --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
  func.func private @mlir_reshape_convolution_real__part_0(%arg0: !migraphx.shaped<1x1x16x1x16x1xf32, 256x256x16x16x1x1> {func.read_access}, %arg1: !migraphx.shaped<1x1x3x3xf32, 9x9x3x1> {func.read_access}) -> (!migraphx.shaped<1x1x32x32xf32, 1024x1024x32x1> {func.write_access}) {
    %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [1, 1, 16, 2, 16, 2]} : <1x1x16x1x16x1xf32, 256x256x16x16x1x1> -> <1x1x16x2x16x2xf32, 256x256x16x0x1x0>
    %1 = migraphx.reshape %0 {dims = [2, 4, 32, 32]} : <1x1x16x2x16x2xf32, 256x256x16x0x1x0> -> <1x1x32x32xf32, 1024x1024x32x1>
    %2 = migraphx.convolution %1, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x1x32x32xf32, 1024x1024x32x1>, <1x1x3x3xf32, 9x9x3x1> -> <1x1x32x32xf32, 1024x1024x32x1>
    return %2 : !migraphx.shaped<1x1x32x32xf32, 1024x1024x32x1>
  }
  func.func @mlir_reshape_convolution_real(%arg0: !migraphx.shaped<1x1x16x1x16x1xf32, 256x256x16x16x1x1>, %arg1: !migraphx.shaped<1x1x3x3xf32, 9x9x3x1>) -> !migraphx.shaped<1x1x32x32xf32, 1024x1024x32x1> {
    %token, %results = mhal.launch @mlir_reshape_convolution_real__part_0 (%arg0, %arg1) : (!migraphx.shaped<1x1x16x1x16x1xf32, 256x256x16x16x1x1>, !migraphx.shaped<1x1x3x3xf32, 9x9x3x1>) -> !migraphx.shaped<1x1x32x32xf32, 1024x1024x32x1>
    mhal.await %token : !mhal.token
    return %results : !migraphx.shaped<1x1x32x32xf32, 1024x1024x32x1>
  }
  module @__xmodule_ attributes {mhal.arch = "gfx908", mhal.module} {
    func.func private @mlir_reshape_convolution_real__part_0(%arg0: !migraphx.shaped<1x1x16x1x16x1xf32, 256x256x16x16x1x1> {func.read_access}, %arg1: !migraphx.shaped<1x1x3x3xf32, 9x9x3x1> {func.read_access}) -> (!migraphx.shaped<1x1x32x32xf32, 1024x1024x32x1> {func.write_access}) attributes {kernel, original_func = @mlir_reshape_convolution_real__part_0} {
      %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [1, 1, 16, 2, 16, 2]} : <1x1x16x1x16x1xf32, 256x256x16x16x1x1> -> <1x1x16x2x16x2xf32, 256x256x16x0x1x0>
      %1 = migraphx.reshape %0 {dims = [2, 4, 32, 32]} : <1x1x16x2x16x2xf32, 256x256x16x0x1x0> -> <1x1x32x32xf32, 1024x1024x32x1>
      %2 = migraphx.convolution %1, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x1x32x32xf32, 1024x1024x32x1>, <1x1x3x3xf32, 9x9x3x1> -> <1x1x32x32xf32, 1024x1024x32x1>
      return %2 : !migraphx.shaped<1x1x32x32xf32, 1024x1024x32x1>
    }
  }
}

