// RUN: sed -e 's/##TOKEN_ARCH##/%arch/g' %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_transpose_reshape_dot_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
  func.func private @mlir_transpose_reshape_dot(%arg0: !migraphx.shaped<2x8x4x4xf32, 128x16x4x1> {func.read_access}, %arg1: !migraphx.shaped<1x8x8xf32, 64x8x1> {func.read_access}) -> (!migraphx.shaped<2x16x8xf32, 128x8x1> {func.write_access}) {
    %0 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [2, 8, 8]} : !migraphx.shaped<1x8x8xf32, 64x8x1> -> !migraphx.shaped<2x8x8xf32, 64x8x1>
    %1 = migraphx.transpose %arg0 {permutation = [0, 2, 3, 1]} : !migraphx.shaped<2x8x4x4xf32, 128x16x4x1> -> !migraphx.shaped<2x4x4x8xf32, 128x32x8x1>
    %2 = migraphx.reshape %1 {dims = [2, 16, 8]} : !migraphx.shaped<2x4x4x8xf32, 128x32x8x1> -> !migraphx.shaped<2x16x8xf32, 128x8x1>
    %3 = migraphx.dot %2, %0 : !migraphx.shaped<2x16x8xf32, 128x8x1>, !migraphx.shaped<2x8x8xf32, 64x8x1> -> !migraphx.shaped<2x16x8xf32, 128x8x1>
    return %3 : !migraphx.shaped<2x16x8xf32, 128x8x1>
  }
  func.func @mlir_transpose_reshape_dot_wrapper(%arg0: !migraphx.shaped<2x8x4x4xf32, 128x16x4x1>, %arg1: !migraphx.shaped<1x8x8xf32, 64x8x1> ) -> !migraphx.shaped<2x16x8xf32, 128x8x1> {
    %token, %results = mhal.launch @mlir_transpose_reshape_dot (%arg0, %arg1) : (!migraphx.shaped<2x8x4x4xf32, 128x16x4x1>, !migraphx.shaped<1x8x8xf32, 64x8x1>) -> !migraphx.shaped<2x16x8xf32, 128x8x1>
    mhal.await %token : !mhal.token
    return %results : !migraphx.shaped<2x16x8xf32, 128x8x1> 
  }
  module @__xmodule_ attributes {mhal.arch = "##TOKEN_ARCH##", mhal.module} {
    func.func private @mlir_transpose_reshape_dot(%arg0: !migraphx.shaped<2x8x4x4xf32, 128x16x4x1> {func.read_access}, %arg1: !migraphx.shaped<1x8x8xf32, 64x8x1> {func.read_access}) -> (!migraphx.shaped<2x16x8xf32, 128x8x1> {func.write_access}) attributes {kernel, original_func = @mlir_transpose_reshape_dot} {
      %0 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [2, 8, 8]} : !migraphx.shaped<1x8x8xf32, 64x8x1> -> !migraphx.shaped<2x8x8xf32, 64x8x1>
      %1 = migraphx.transpose %arg0 {permutation = [0, 2, 3, 1]} : !migraphx.shaped<2x8x4x4xf32, 128x16x4x1> -> !migraphx.shaped<2x4x4x8xf32, 128x32x8x1>
      %2 = migraphx.reshape %1 {dims = [2, 16, 8]} : !migraphx.shaped<2x4x4x8xf32, 128x32x8x1> -> !migraphx.shaped<2x16x8xf32, 128x8x1>
      %3 = migraphx.dot %2, %0 : !migraphx.shaped<2x16x8xf32, 128x8x1>, !migraphx.shaped<2x8x8xf32, 64x8x1> -> !migraphx.shaped<2x16x8xf32, 128x8x1>
      return %3 : !migraphx.shaped<2x16x8xf32, 128x8x1>
    }
  }
}

