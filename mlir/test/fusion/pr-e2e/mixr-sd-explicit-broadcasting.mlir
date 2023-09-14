// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | rocmlir-driver -host-pipeline=partition,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_reshape_convolution --verifier clone - | rocmlir-driver -c -arch %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
// Note the fake wrapper function to make it look like this has already been partitioned,
// since we want "partition" for the xmodel packaging but don't actually want to use
// the partition logic.
// This is an awkward hack that should be fixed Later (tm)
module {
  func.func @mlir_reshape_convolution_real(%arg0: tensor<1x1x16x1x16x1xf32>, %arg1: tensor<1x1x3x3xf32>) -> tensor<1x1x32x32xf32> attributes {kernel = "mixr", num_cu = 48 : i64} {
    %0 = migraphx.multibroadcast(%arg0) {out_dyn_dims = [], out_lens = [1, 1, 16, 2, 16, 2]} : (tensor<1x1x16x1x16x1xf32>) -> tensor<1x1x16x2x16x2xf32>
    %1 = migraphx.reshape(%0) {dims = [2, 4, 32, 32]} : (tensor<1x1x16x2x16x2xf32>) -> tensor<1x1x32x32xf32>
    %2 = migraphx.convolution(%1, %arg1) {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<1x1x32x32xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x32x32xf32>
    return %2 : tensor<1x1x32x32xf32>
  }

  func.func @mlir_reshape_convolution(%arg0: tensor<1x1x16x1x16x1xf32>, %arg1: tensor<1x1x3x3xf32>) -> tensor<1x1x32x32xf32> {
    %ret = call @mlir_reshape_convolution_real(%arg0, %arg1) : (tensor<1x1x16x1x16x1xf32>, tensor<1x1x3x3xf32>) -> (tensor<1x1x32x32xf32>)
    return %ret : tensor<1x1x32x32xf32>
  }
}

