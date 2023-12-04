// RUN: sed -e s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.000004 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1> {func.read_access}, %arg1: !migraphx.shaped<1x12x64x384xf32, 294912x24576x384x1> {func.read_access}, %arg2: !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1> {func.read_access}) -> (!migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1> {func.write_access}) {
    %0 = migraphx.dot %arg0, %arg1: <1x12x384x64xf32, 294912x24576x64x1>, <1x12x64x384xf32, 294912x24576x384x1> -> <1x12x384x384xf32, 1769472x147456x384x1>
    %1 = migraphx.softmax %0{axis = 3 : i64} : <1x12x384x384xf32, 1769472x147456x384x1> -> <1x12x384x384xf32, 1769472x147456x384x1>
    %2 = migraphx.dot %1, %arg2: <1x12x384x384xf32, 1769472x147456x384x1>, <1x12x384x64xf32, 294912x24576x64x1> -> <1x12x384x64xf32, 294912x24576x64x1>
    return %2 : !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>
  }
  func.func @mlir_attention_wrapper(%arg0: !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>, %arg1: !migraphx.shaped<1x12x64x384xf32, 294912x24576x384x1>,  %arg2: !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>) -> !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1> {
    %token, %results = mhal.launch @mlir_attention (%arg0, %arg1, %arg2) : (!migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>, !migraphx.shaped<1x12x64x384xf32, 294912x24576x384x1>, !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>) -> !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>
    mhal.await %token : !mhal.token
    return %results : !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>
  }
  module @__xmodule_ attributes {mhal.arch = "##TOKEN_ARCH##", mhal.module} {
    func.func private @mlir_attention(%arg0: !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1> {func.read_access}, %arg1: !migraphx.shaped<1x12x64x384xf32, 294912x24576x384x1> {func.read_access}, %arg2: !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1> {func.read_access}) -> (!migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1> {func.write_access}) attributes {kernel, original_func = @mlir_attention} {
      %0 = migraphx.dot %arg0, %arg1: <1x12x384x64xf32, 294912x24576x64x1>, <1x12x64x384xf32, 294912x24576x384x1> -> <1x12x384x384xf32, 1769472x147456x384x1>
      %1 = migraphx.softmax %0{axis = 3 : i64} : <1x12x384x384xf32, 1769472x147456x384x1> -> <1x12x384x384xf32, 1769472x147456x384x1>
      %2 = migraphx.dot %1, %arg2: <1x12x384x384xf32, 1769472x147456x384x1>, <1x12x384x64xf32, 294912x24576x64x1> -> <1x12x384x64xf32, 294912x24576x64x1>
      return %2 : !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>
    }
  }
}
