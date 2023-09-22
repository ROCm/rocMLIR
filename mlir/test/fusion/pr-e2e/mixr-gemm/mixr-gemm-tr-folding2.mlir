// RUN: sed -e 's/##TOKEN_ARCH##/%arch/g' %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_transpose_reshape_dot_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
  func.func @mlir_transpose_reshape_dot(%arg0: tensor<1x2x1x3xf32> {func.read_access}, %arg1: tensor<6x6xf32> {func.read_access}) -> (tensor<1x6xf32> {func.write_access}) {
    %0 = migraphx.transpose(%arg0) {permutation = [0, 2, 1, 3]} : (tensor<1x2x1x3xf32>) -> tensor<1x1x2x3xf32>
    %1 = migraphx.reshape(%0) {dims = [1, 6]} : (tensor<1x1x2x3xf32>) -> tensor<1x6xf32>
    %2 = migraphx.dot(%1, %arg1) : (tensor<1x6xf32>, tensor<6x6xf32>) -> tensor<1x6xf32>
    return %2 : tensor<1x6xf32>
  }
  func.func @mlir_transpose_reshape_dot_wrapper(%arg0: tensor<1x2x1x3xf32>, %arg1: tensor<6x6xf32>) -> tensor<1x6xf32> {
    %token, %results = mhal.launch @mlir_transpose_reshape_dot (%arg0, %arg1) : (tensor<1x2x1x3xf32>, tensor<6x6xf32>) -> tensor<1x6xf32>
    mhal.await %token : !mhal.token
    return %results : tensor<1x6xf32>
  }
  module @__xmodule_ attributes {mhal.arch = "##TOKEN_ARCH##", mhal.module} {
    func.func @mlir_transpose_reshape_dot(%arg0: tensor<1x2x1x3xf32> {func.read_access}, %arg1: tensor<6x6xf32> {func.read_access}) -> (tensor<1x6xf32> {func.write_access}) attributes {kernel, original_func = @mlir_transpose_reshape_dot} {
      %0 = migraphx.transpose(%arg0) {permutation = [0, 2, 1, 3]} : (tensor<1x2x1x3xf32>) -> tensor<1x1x2x3xf32>
      %1 = migraphx.reshape(%0) {dims = [1, 6]} : (tensor<1x1x2x3xf32>) -> tensor<1x6xf32>
      %2 = migraphx.dot(%1, %arg1) : (tensor<1x6xf32>, tensor<6x6xf32>) -> tensor<1x6xf32>
      return %2 : tensor<1x6xf32>
    }
  }
}

