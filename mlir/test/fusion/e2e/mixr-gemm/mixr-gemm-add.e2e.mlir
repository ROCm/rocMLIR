// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -c  | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {
  // CHECK:  [5,     5,     5],
  // CHECK-NEXT: [5,     5,     5],
  // CHECK-NEXT: [5,     5,     5],
  // CHECK-NEXT: [5,     5,     5],
  // CHECK-NEXT: [5,     5,     5]
  func.func @dot_add(%arg0: tensor<1x5x4xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x5x3xf32>) -> tensor<1x5x3xf32> attributes{kernel, arch = ""} {
    %0 = "migraphx.dot"(%arg0, %arg1) : (tensor<1x5x4xf32>, tensor<1x4x3xf32>) -> tensor<1x5x3xf32>
    %2 = "migraphx.add"(%0, %arg2) {} : (tensor<1x5x3xf32>, tensor<1x5x3xf32>)-> tensor<1x5x3xf32>
    return %2 : tensor<1x5x3xf32>
  }
}
