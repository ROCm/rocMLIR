// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
module {
  // CHECK:  {{.*}}[4, 4, 4, 4],
  // CHECK-NEXT:   [4, 4, 4, 4]{{.*}}
  func.func @mlir_dot(%arg0: tensor<1x2x4xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x4x3xf32>) -> tensor<1x2x4xf32> attributes{kernel, arch = ""} {
    %0 = migraphx.multibroadcast(%arg1) {out_dyn_dims = [], out_lens = [1, 2, 3]} : (tensor<1x1x1xf32>) -> tensor<1x2x3xf32>
    %arg2tp = migraphx.transpose(%arg2) {permutation = [0:i64, 2:i64, 1:i64]} : (tensor<1x4x3xf32>)-> tensor<1x3x4xf32>
    %1 = migraphx.dot(%0, %arg2tp) : (tensor<1x2x3xf32>, tensor<1x3x4xf32>) -> tensor<1x2x4xf32>
    %2 = migraphx.add(%1, %arg0) : (tensor<1x2x4xf32>, tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
    return %2 : tensor<1x2x4xf32>
  }
}
