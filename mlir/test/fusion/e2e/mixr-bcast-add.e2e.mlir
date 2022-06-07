// RUN: miopen-opt -migraphx-to-tosa %s | mlir-miopen-driver -host-pipeline highlevel | miopen-gen -ph -print-results -rand none - | mlir-miopen-driver -c  | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK:  6,     6,     6,     6
module {
  func @main0(%arg0: tensor<1x5x4x4xf32>, %arg1: tensor<4x5x1x1xf32>, %arg2: tensor<4xf32>) -> tensor<1x4x4x4xf32> attributes{kernel} {
    %0 = "migraphx.convolution"(%arg0, %arg1) {padding = [0:i64, 0:i64, 0:i64, 0:i64], stride = [1:i64, 1:i64], dilation = [1:i64, 1:i64], group = 1:i64} : (tensor<1x5x4x4xf32>, tensor<4x5x1x1xf32>) -> tensor<1x4x4x4xf32>
    %1 = "migraphx.broadcast"(%arg2) {axis = 1:i64, broadcast_lens= [1:i64, 4:i64, 4:i64, 4:i64] } : (tensor<4xf32>)-> tensor<1x4x4x4xf32>
    %2 = "migraphx.add"(%0, %1) {} : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>)-> tensor<1x4x4x4xf32>
    return %2 : tensor<1x4x4x4xf32>
  }
}
