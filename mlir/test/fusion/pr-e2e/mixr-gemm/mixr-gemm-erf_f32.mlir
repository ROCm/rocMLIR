// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -print-results -rand 3 - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
module {
  // CHECK:  [-1,     1,     1],
  // CHECK-NEXT: [-1,     0.999978,     -1],
  // CHECK-NEXT: [-1,     1,     -1],
  // CHECK-NEXT: [0,     -1,     -0.999978],
  // CHECK-NEXT: [0,     -1,     -1]
  func.func @dot_add(%arg0: tensor<1x5x4xf32>, %arg1: tensor<1x4x3xf32>) -> tensor<1x5x3xf32> attributes{kernel, arch = ""} {
    %0 = "migraphx.dot"(%arg0, %arg1) : (tensor<1x5x4xf32>, tensor<1x4x3xf32>) -> tensor<1x5x3xf32>
    %2 = "migraphx.erf"(%0) : (tensor<1x5x3xf32>)-> tensor<1x5x3xf32>
    return %2 : tensor<1x5x3xf32>
  }
}
