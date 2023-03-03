// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch %s | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut dot_add --verifier clone - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  func.func @dot_add(%arg0: tensor<1x128x64xf32>, %arg1: tensor<1x64x256xf32>) -> tensor<1x128x1xf32> {
    %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x128x64xf32>, tensor<1x64x256xf32>) -> tensor<1x128x256xf32>
    %1 = "tosa.reduce_sum"(%0) {axis = 2 : i64} : (tensor<1x128x256xf32>) -> tensor<1x128x1xf32>
    return %1 : tensor<1x128x1xf32>
  }
}
