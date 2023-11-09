// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_convolution_add_clip_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
  func.func private @mlir_convolution_add_clip(%arg0: tensor<1x4x1x1xf32> {func.read_access}, %arg1: tensor<4x3x3x3xf32> {func.read_access}, %arg2: tensor<4x3x3x3xf32> {func.read_access}) -> (tensor<4x4x1x1xf32> {func.write_access}) {
    %0 = migraphx.multibroadcast(%arg0) {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : (tensor<1x4x1x1xf32>) -> tensor<4x4x1x1xf32>
    %1 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = migraphx.multibroadcast(%2) {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : (tensor<1xf32>) -> tensor<4x4x1x1xf32>
    %4 = migraphx.multibroadcast(%1) {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : (tensor<1xf32>) -> tensor<4x4x1x1xf32>
    %5 = migraphx.convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<4x3x3x3xf32>, tensor<4x3x3x3xf32>) -> tensor<4x4x1x1xf32>
    %6 = migraphx.add(%5, %0) : (tensor<4x4x1x1xf32>, tensor<4x4x1x1xf32>) -> tensor<4x4x1x1xf32>
    %7 = migraphx.clip(%6, %3, %4) : (tensor<4x4x1x1xf32>, tensor<4x4x1x1xf32>, tensor<4x4x1x1xf32>) -> tensor<4x4x1x1xf32>
    return %7 : tensor<4x4x1x1xf32>
  }
  func.func @mlir_convolution_add_clip_wrapper(%arg0: tensor<1x4x1x1xf32>, %arg1: tensor<4x3x3x3xf32>, %arg2: tensor<4x3x3x3xf32>) -> tensor<4x4x1x1xf32> {
    %token, %results = mhal.launch @mlir_convolution_add_clip (%arg0, %arg1, %arg2) : (tensor<1x4x1x1xf32>, tensor<4x3x3x3xf32>, tensor<4x3x3x3xf32>) -> tensor<4x4x1x1xf32>
    mhal.await %token : !mhal.token
    return %results : tensor<4x4x1x1xf32>
  }
  module @__xmodule_ attributes {mhal.arch = "##TOKEN_ARCH##", mhal.module} {
    func.func private @mlir_convolution_add_clip(%arg0: tensor<1x4x1x1xf32> {func.read_access}, %arg1: tensor<4x3x3x3xf32> {func.read_access}, %arg2: tensor<4x3x3x3xf32> {func.read_access}) -> (tensor<4x4x1x1xf32> {func.write_access}) attributes {kernel, mhal.reference_func = @mlir_convolution_add_clip} {
      %0 = migraphx.multibroadcast(%arg0) {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : (tensor<1x4x1x1xf32>) -> tensor<4x4x1x1xf32>
      %1 = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
      %2 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
      %3 = migraphx.multibroadcast(%2) {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : (tensor<1xf32>) -> tensor<4x4x1x1xf32>
      %4 = migraphx.multibroadcast(%1) {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : (tensor<1xf32>) -> tensor<4x4x1x1xf32>
      %5 = migraphx.convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<4x3x3x3xf32>, tensor<4x3x3x3xf32>) -> tensor<4x4x1x1xf32>
      %6 = migraphx.add(%5, %0) : (tensor<4x4x1x1xf32>, tensor<4x4x1x1xf32>) -> tensor<4x4x1x1xf32>
      %7 = migraphx.clip(%6, %3, %4) : (tensor<4x4x1x1xf32>, tensor<4x4x1x1xf32>, tensor<4x4x1x1xf32>) -> tensor<4x4x1x1xf32>
      return %7 : tensor<4x4x1x1xf32>
    }
  }
}

