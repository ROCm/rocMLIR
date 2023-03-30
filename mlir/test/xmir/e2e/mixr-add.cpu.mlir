// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -print-results -fut main0 -verifier clone - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext,%linalg_test_lib_dir/%prefix_mlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%prefix_mlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2


// CHECK:  [1 1 1]
module {
  func.func @main0(%arg0: tensor<1x5x4x4xf32>, %arg1: tensor<4x5x1x1xf32>, %arg2: tensor<4xf32>) -> tensor<1x4x4x4xf32> {
    %0 = "migraphx.convolution"(%arg0, %arg1) {padding = [0:i64, 0:i64, 0:i64, 0:i64], stride = [1:i64, 1:i64], dilation = [1:i64, 1:i64], group = 1:i64} : (tensor<1x5x4x4xf32>, tensor<4x5x1x1xf32>) -> tensor<1x4x4x4xf32>
    %1 = "migraphx.broadcast"(%arg2) {axis = 1:i64, out_lens= [1:i64, 4:i64, 4:i64, 4:i64] } : (tensor<4xf32>)-> tensor<1x4x4x4xf32>
    %2 = "migraphx.add"(%0, %1) {} : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>)-> tensor<1x4x4x4xf32>
    return %2 : tensor<1x4x4x4xf32>
  }
}
