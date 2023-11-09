// RUN: sed -e s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.02 -absDiff_threshold 0.02 -RMS_threshold 0.01 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
  func.func private @mlir_attention(%arg0: tensor<1x12x384x64xf16> {func.read_access}, %arg1: tensor<1x12x64x384xf16> {func.read_access}, %arg2: tensor<1x12x384x64xf16> {func.read_access}) -> (tensor<1x12x384x64xf16> {func.write_access}) {
    %0 = migraphx.dot(%arg0, %arg1): (tensor<1x12x384x64xf16>, tensor<1x12x64x384xf16>) -> tensor<1x12x384x384xf16>
    %1 = migraphx.softmax(%0){axis = 3 : i64} : tensor<1x12x384x384xf16> -> tensor<1x12x384x384xf16>
    %2 = migraphx.dot(%1, %arg2): (tensor<1x12x384x384xf16>, tensor<1x12x384x64xf16>) -> tensor<1x12x384x64xf16>
    return %2 : tensor<1x12x384x64xf16>
  }
  func.func @mlir_attention_wrapper(%arg0: tensor<1x12x384x64xf16>, %arg1: tensor<1x12x64x384xf16>,  %arg2: tensor<1x12x384x64xf16>) -> tensor<1x12x384x64xf16> {
    %token, %results = mhal.launch @mlir_attention (%arg0, %arg1, %arg2) : (tensor<1x12x384x64xf16>, tensor<1x12x64x384xf16>, tensor<1x12x384x64xf16>) -> tensor<1x12x384x64xf16>
    mhal.await %token : !mhal.token
    return %results : tensor<1x12x384x64xf16>
  }
  module @__xmodule_ attributes {mhal.arch = "##TOKEN_ARCH##", mhal.module} {
    func.func private @mlir_attention(%arg0: tensor<1x12x384x64xf16> {func.read_access}, %arg1: tensor<1x12x64x384xf16> {func.read_access}, %arg2: tensor<1x12x384x64xf16> {func.read_access}) -> (tensor<1x12x384x64xf16> {func.write_access}) attributes {kernel, mhal.reference_func = @mlir_attention} {
      %0 = migraphx.dot(%arg0, %arg1): (tensor<1x12x384x64xf16>, tensor<1x12x64x384xf16>) -> tensor<1x12x384x384xf16>
      %1 = migraphx.softmax(%0){axis = 3 : i64} : tensor<1x12x384x384xf16> -> tensor<1x12x384x384xf16>
      %2 = migraphx.dot(%1, %arg2): (tensor<1x12x384x384xf16>, tensor<1x12x384x64xf16>) -> tensor<1x12x384x64xf16>
      return %2 : tensor<1x12x384x64xf16>
    }
  }
}
