// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut quant_dot --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2

module {
  // CHECK:  [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  func.func @quant_dot(%arg0: tensor<1x5x4xi8>, %arg1: tensor<1x4x3xi8>) -> tensor<1x15xi32> attributes{kernel, arch = ""} {
    %0 = migraphx.quant_dot(%arg0, %arg1) : (tensor<1x5x4xi8>, tensor<1x4x3xi8>) -> tensor<1x5x3xi32>
    %1 = migraphx.reshape(%0) {dims = [1:i64, 15:i64]} : (tensor<1x5x3xi32>)-> tensor<1x15xi32>
    return %1 : tensor<1x15xi32>
  }
}
