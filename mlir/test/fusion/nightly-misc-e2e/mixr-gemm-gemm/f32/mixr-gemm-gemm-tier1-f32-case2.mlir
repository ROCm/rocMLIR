// RUN: rocmlir-gen -fut mlir_gemm_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_gemm_gemm_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func private @mlir_gemm_gemm(%arg0: !migraphx.shaped<2x16x384x64xf32, 393216x24576x64x1>, %arg1: !migraphx.shaped<2x16x64x384xf32, 393216x24576x384x1>, %arg2: !migraphx.shaped<2x16x384x64xf32, 393216x24576x64x1>) -> (!migraphx.shaped<2x16x384x64xf32, 393216x24576x64x1>) {
    %0 = migraphx.dot %arg0, %arg1: <2x16x384x64xf32, 393216x24576x64x1>, <2x16x64x384xf32, 393216x24576x384x1> -> <2x16x384x384xf32, 2359296x147456x384x1>
    %1 = migraphx.dot %0, %arg2: <2x16x384x384xf32, 2359296x147456x384x1>, <2x16x384x64xf32, 393216x24576x64x1> -> <2x16x384x64xf32, 393216x24576x64x1>
    return %1 : !migraphx.shaped<2x16x384x64xf32, 393216x24576x64x1>
  }
}
