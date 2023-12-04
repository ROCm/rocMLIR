// RUN: sed -e 's/##TOKEN_ARCH##/%arch/g' %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_transpose_reshape_dot_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
  func.func @mlir_transpose_reshape_dot(%arg0: !migraphx.shaped<1x2x1x3xf32, 6x3x3x1> {func.read_access}, %arg1: !migraphx.shaped<6x6xf32, 6x1> {func.read_access}) -> (!migraphx.shaped<1x6xf32, 6x1> {func.write_access}) {
    %0 = migraphx.transpose %arg0 {permutation = [0, 2, 1, 3]} : <1x2x1x3xf32, 6x3x3x1> -> <1x1x2x3xf32, 6x6x3x1>
    %1 = migraphx.reshape %0 {dims = [1, 6]} : <1x1x2x3xf32, 6x6x3x1> -> <1x6xf32, 6x1>
    %2 = migraphx.dot %1, %arg1 : <1x6xf32, 6x1>, <6x6xf32, 6x1> -> <1x6xf32, 6x1>
    return %2 : !migraphx.shaped<1x6xf32, 6x1>
  }
  func.func @mlir_transpose_reshape_dot_wrapper(%arg0: !migraphx.shaped<1x2x1x3xf32, 6x3x3x1>, %arg1: !migraphx.shaped<6x6xf32, 6x1>) -> !migraphx.shaped<1x6xf32, 6x1> {
    %token, %results = mhal.launch @mlir_transpose_reshape_dot (%arg0, %arg1) : (!migraphx.shaped<1x2x1x3xf32, 6x3x3x1>, !migraphx.shaped<6x6xf32, 6x1>) -> !migraphx.shaped<1x6xf32, 6x1>
    mhal.await %token : !mhal.token
    return %results : !migraphx.shaped<1x6xf32, 6x1>
  }
  module @__xmodule_ attributes {mhal.arch = "##TOKEN_ARCH##", mhal.module} {
    func.func @mlir_transpose_reshape_dot(%arg0: !migraphx.shaped<1x2x1x3xf32, 6x3x3x1> {func.read_access}, %arg1: !migraphx.shaped<6x6xf32, 6x1> {func.read_access}) -> (!migraphx.shaped<1x6xf32, 6x1> {func.write_access}) attributes {kernel, original_func = @mlir_transpose_reshape_dot} {
      %0 = migraphx.transpose %arg0 {permutation = [0, 2, 1, 3]} : <1x2x1x3xf32, 6x3x3x1> -> <1x1x2x3xf32, 6x6x3x1>
      %1 = migraphx.reshape %0 {dims = [1, 6]} : <1x1x2x3xf32, 6x6x3x1> -> <1x6xf32, 6x1>
      %2 = migraphx.dot %1, %arg1 : <1x6xf32, 6x1>, <6x6xf32, 6x1> -> <1x6xf32, 6x1>
      return %2 : !migraphx.shaped<1x6xf32, 6x1>
    }
  }
}

