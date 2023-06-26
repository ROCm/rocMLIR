// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut dot_add --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base

module {
  func.func private @dot_add__part_0(%arg0: tensor<1x128x64xf32> {func.read_access}, %arg1: tensor<1x64x256xf32> {func.read_access}, %arg2: tensor<1x128x256xf32> {func.read_access}) -> (tensor<1x128x1xf32> {func.write_access}) {
    %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x128x64xf32>, tensor<1x64x256xf32>) -> tensor<1x128x256xf32>
    %1 = "tosa.add"(%0, %arg2) : (tensor<1x128x256xf32>, tensor<1x128x256xf32>) -> tensor<1x128x256xf32>
    %2 = "tosa.reduce_sum"(%1) {axis = 2 : i64} : (tensor<1x128x256xf32>) -> tensor<1x128x1xf32>
    return %2 : tensor<1x128x1xf32>
  }
  func.func @dot_add(%arg0: tensor<1x128x64xf32>, %arg1: tensor<1x64x256xf32>, %arg2: tensor<1x128x256xf32>) -> tensor<1x128x1xf32> {
    %token, %results = mhal.launch @dot_add__part_0 (%arg0, %arg1, %arg2) : (tensor<1x128x64xf32>, tensor<1x64x256xf32>, tensor<1x128x256xf32>) -> tensor<1x128x1xf32>
    mhal.await %token : !mhal.token
    return %results : tensor<1x128x1xf32>
  }
  module @__xmodule_ attributes {mhal.arch = "##TOKEN_ARCH##", mhal.module} {
    func.func private @dot_add__part_0(%arg0: tensor<1x128x64xf32> {func.read_access}, %arg1: tensor<1x64x256xf32> {func.read_access}, %arg2: tensor<1x128x256xf32> {func.read_access}) -> (tensor<1x128x1xf32> {func.write_access}) attributes {kernel, original_func = @dot_add__part_0} {
      %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x128x64xf32>, tensor<1x64x256xf32>) -> tensor<1x128x256xf32>
      %1 = "tosa.add"(%0, %arg2) : (tensor<1x128x256xf32>, tensor<1x128x256xf32>) -> tensor<1x128x256xf32>
      %2 = "tosa.reduce_sum"(%1) {axis = 2 : i64} : (tensor<1x128x256xf32>) -> tensor<1x128x1xf32>
      return %2 : tensor<1x128x1xf32>
    }
  }
}
