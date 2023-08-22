// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_convolution_add_clip_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
  func.func private @mlir_convolution_add_clip(%arg0: !migraphx.shaped<1x4x1x1xf32, 4x1x1x1> {func.read_access}, %arg1: !migraphx.shaped<4x3x3x3xf32, 27x9x3x1> {func.read_access}, %arg2: !migraphx.shaped<4x3x3x3xf32, 27x9x3x1> {func.read_access}) -> (!migraphx.shaped<4x4x1x1xf32, 4x1x1x1> {func.write_access}) {
    %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : !migraphx.shaped<1x4x1x1xf32, 4x1x1x1> -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
    %1 = migraphx.literal (dense<6.000000e+00> : tensor<1xf32>) : !migraphx.shaped<1xf32, 1>
    %2 = migraphx.literal (dense<0.000000e+00> : tensor<1xf32>) : !migraphx.shaped<1xf32, 1>
    %3 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : !migraphx.shaped<1xf32, 1> -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
    %4 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : !migraphx.shaped<1xf32, 1> -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
    %5 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : !migraphx.shaped<4x3x3x3xf32, 27x9x3x1>, !migraphx.shaped<4x3x3x3xf32, 27x9x3x1> -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
    %6 = migraphx.add %5, %0 : !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>, !migraphx.shaped<4x4x1x1xf32, 4x1x1x1> -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
    %7 = migraphx.clip %6, %3, %4 : !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>, !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>, !migraphx.shaped<4x4x1x1xf32, 4x1x1x1> -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
    return %7 : !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
  }
  func.func @mlir_convolution_add_clip_wrapper(%arg0: !migraphx.shaped<1x4x1x1xf32, 4x1x1x1>, %arg1: !migraphx.shaped<4x3x3x3xf32, 27x9x3x1>, %arg2: !migraphx.shaped<4x3x3x3xf32, 27x9x3x1>) -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1> {
    %token, %results = mhal.launch @mlir_convolution_add_clip (%arg0, %arg1, %arg2) : (!migraphx.shaped<1x4x1x1xf32, 4x1x1x1>, !migraphx.shaped<4x3x3x3xf32, 27x9x3x1>, !migraphx.shaped<4x3x3x3xf32, 27x9x3x1>) -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
    mhal.await %token : !mhal.token
    return %results : !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
  }
  module @__xmodule_ attributes {mhal.arch = "##TOKEN_ARCH##", mhal.module} {
    func.func private @mlir_convolution_add_clip(%arg0: !migraphx.shaped<1x4x1x1xf32, 4x1x1x1> {func.read_access}, %arg1: !migraphx.shaped<4x3x3x3xf32, 27x9x3x1> {func.read_access}, %arg2: !migraphx.shaped<4x3x3x3xf32, 27x9x3x1> {func.read_access}) -> (!migraphx.shaped<4x4x1x1xf32, 4x1x1x1> {func.write_access}) attributes {kernel, original_func = @mlir_convolution_add_clip} {
      %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : !migraphx.shaped<1x4x1x1xf32, 4x1x1x1> -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
      %1 = migraphx.literal (dense<6.000000e+00> : tensor<1xf32>) : !migraphx.shaped<1xf32, 1>
      %2 = migraphx.literal (dense<0.000000e+00> : tensor<1xf32>) : !migraphx.shaped<1xf32, 1>
      %3 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : !migraphx.shaped<1xf32, 1> -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
      %4 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : !migraphx.shaped<1xf32, 1> -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
      %5 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : !migraphx.shaped<4x3x3x3xf32, 27x9x3x1>, !migraphx.shaped<4x3x3x3xf32, 27x9x3x1> -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
      %6 = migraphx.add %5, %0 : !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>, !migraphx.shaped<4x4x1x1xf32, 4x1x1x1> -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
      %7 = migraphx.clip %6, %3, %4 : !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>, !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>, !migraphx.shaped<4x4x1x1xf32, 4x1x1x1> -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
      return %7 : !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
    }
  }
}

