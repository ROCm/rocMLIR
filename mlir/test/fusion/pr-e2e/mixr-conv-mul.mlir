// RUN: rocmlir-opt --migraphx-transform --canonicalize --migraphx-to-tosa %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_convolution --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_convolution(%arg0: tensor<1x2048x1x1xf32>, %arg1: tensor<1x2048x1x1xf32>, %arg2: tensor<1x2048x7x7xf32>, %arg3: tensor<1x2048x1x1xf32>, %arg4: tensor<1x1024x14x14xf32>, %arg5: tensor<2048x1024x1x1xf32>) -> tensor<1x2048x7x7xf32> {
    %0 = migraphx.multibroadcast(%arg3) {out_dyn_dims = [], out_lens = [1, 2048, 7, 7]} : (tensor<1x2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %1 = migraphx.multibroadcast(%arg1) {out_dyn_dims = [], out_lens = [1, 2048, 7, 7]} : (tensor<1x2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %2 = migraphx.multibroadcast(%arg0) {out_dyn_dims = [], out_lens = [1, 2048, 7, 7]} : (tensor<1x2048x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %3 = migraphx.convolution(%arg4, %arg5) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [2, 2]} : (tensor<1x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %4 = migraphx.mul(%2, %3) : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %5 = migraphx.mul(%1, %4) : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %6 = migraphx.mul(%2, %arg2) : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %7 = migraphx.mul(%1, %6) : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %8 = migraphx.add(%7, %5) : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %9 = migraphx.add(%8, %0) : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %10 = migraphx.relu(%9) : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    return %10 : tensor<1x2048x7x7xf32>
  }
}
