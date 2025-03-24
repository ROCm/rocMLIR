// RUN: rocmlir-gen -fut mlir_gemm_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_gemm_gemm_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
  func.func private @mlir_gemm_gemm(%arg0: !migraphx.shaped<1x4x32x32xf32, 4096x1024x32x1>, 
                            %arg1: !migraphx.shaped<1x4x32x32xf32, 4096x1024x32x1>, 
                            %arg2: !migraphx.shaped<1x4x32x32xf32, 4096x1024x32x1>, 
                            %arg3: !migraphx.shaped<1x4x32x32xf32, 4096x1024x32x1>) 
                            -> (!migraphx.shaped<1x4x32x32xf32, 4096x1024x32x1>) {
    %0 = migraphx.transpose %arg3 {permutation = [0, 1, 3, 2]} : <1x4x32x32xf32, 4096x1024x32x1> -> <1x4x32x32xf32, 4096x1024x32x1>
    %1 = migraphx.dot %arg2, %0 : <1x4x32x32xf32, 4096x1024x32x1>, <1x4x32x32xf32, 4096x1024x32x1> -> <1x4x32x32xf32, 4096x1024x32x1>
    %2 = migraphx.mul %1, %arg1 : <1x4x32x32xf32, 4096x1024x32x1>, <1x4x32x32xf32, 4096x1024x32x1> -> <1x4x32x32xf32, 4096x1024x32x1>
    %3 = migraphx.dot %2, %arg0 : <1x4x32x32xf32, 4096x1024x32x1>, <1x4x32x32xf32, 4096x1024x32x1> -> <1x4x32x32xf32, 4096x1024x32x1>
    return %3 : !migraphx.shaped<1x4x32x32xf32, 4096x1024x32x1>
  }
}
