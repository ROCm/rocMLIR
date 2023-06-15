// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {
  // CHECK: {{.*}}rank = 4 offset = 0 sizes = [1, 4, 2, 2]{{.*}}
  func.func @mlir_quantization(%arg0: tensor<1x4x1x1xf32>, %arg1: tensor<1x4x2x2xi8>, %arg2: tensor<4x4x1x1xi8>) -> tensor<1x4x2x2xi8> attributes {arch = "", kernel = "mixr"} {
    %0 = "tosa.const"() {value = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %1 = "tosa.const"() {value = dense<7.812500e-03> : tensor<1xf32>} : () -> tensor<1xf32>
    %2 = "tosa.const"() {value = dense<1.22070313E-4> : tensor<1xf32>} : () -> tensor<1xf32>
    %3 = migraphx.multibroadcast(%arg0) {out_dyn_dims = [], out_lens = [1, 64, 56, 56]} : (tensor<1x4x1x1xf32>) -> tensor<1x4x2x2xf32>
    %4 = "tosa.reshape"(%0) {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<1xi8>) -> tensor<1x1x1x1xi8>
    %5 = migraphx.multibroadcast(%0) {out_dyn_dims = [], out_lens = [1, 4, 2, 2]} : (tensor<1xi8>) -> tensor<1x4x2x2xi8>
    %6 = migraphx.multibroadcast(%1) {out_dyn_dims = [], out_lens = [1, 4, 2, 2]} : (tensor<1xf32>) -> tensor<1x4x2x2xf32>
    %7 = migraphx.multibroadcast(%2) {out_dyn_dims = [], out_lens = [1, 4, 2, 2]} : (tensor<1xf32>) -> tensor<1x4x2x2xf32>
    %8 = migraphx.quant_convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<1x4x2x2xi8>, tensor<4x4x1x1xi8>) -> tensor<1x4x2x2xi32>
    %9 = migraphx.dequantizelinear(%8, %7) : (tensor<1x4x2x2xi32>, tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
    %10 = migraphx.add(%9, %arg0) : (tensor<1x4x2x2xf32>, tensor<1x4x1x1xf32>) -> tensor<1x4x2x2xf32>
    %11 = migraphx.relu(%10) : (tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
    %12 = migraphx.quantizelinear(%11, %6, %5) : (tensor<1x4x2x2xf32>, tensor<1x4x2x2xf32>, tensor<1x4x2x2xi8>) -> tensor<1x4x2x2xi8>
    return %12 : tensor<1x4x2x2xi8>
  }
}
