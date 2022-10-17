// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut dot_reshape_2 --verifier clone - | rocmlir-driver -host-pipeline xmodel,runner -kernel-pipeline full -targets %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE

module {
  // CHECK:  [4, 4, 4, 4, 4],
  // CHECK-NEXT: [4, 4, 4, 4, 4],
  // CHECK-NEXT: [4, 4, 4, 4, 4]
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  func.func @dot_reshape_2(%arg0: tensor<1x5x4xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x5x3xf32>) -> tensor<1x3x5xf32> attributes{kernel, arch = ""} {
    %0 = "migraphx.dot"(%arg0, %arg1) : (tensor<1x5x4xf32>, tensor<1x4x3xf32>) -> tensor<1x5x3xf32>
    %2 = "migraphx.reshape"(%0) {dims = [1:i64, 3:i64, 5:i64]} : (tensor<1x5x3xf32>)-> tensor<1x3x5xf32>
    return %2 : tensor<1x3x5xf32>
  }
}
