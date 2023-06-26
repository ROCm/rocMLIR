// RUN: rocmlir-opt --migraphx-transform --canonicalize --migraphx-to-tosa %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_dot --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_dot(%arg0: tensor<1x1x1x1xf32>, %arg1: tensor<1x384x2304xf32>, %arg2: tensor<1x384x2304xf32>) -> tensor<1x12x384x384xf32> {
    %0 = "tosa.const"() {value = dense<1.250000e-01> : tensor<1xf32>} : () -> tensor<1xf32>
    %1 = migraphx.multibroadcast(%arg0) {out_dyn_dims = [], out_lens = [1, 12, 384, 384]} : (tensor<1x1x1x1xf32>) -> tensor<1x12x384x384xf32>
    %2 = migraphx.reshape(%arg1) {dims = [1, 384, 36, 64]} : (tensor<1x384x2304xf32>) -> tensor<1x384x36x64xf32>
    %3 = migraphx.transpose(%2) {permutation = [0, 2, 1, 3]} : (tensor<1x384x36x64xf32>) -> tensor<1x36x384x64xf32>
    %4 = migraphx.slice(%3) {axes = [1], ends = [12], starts = [0]} : (tensor<1x36x384x64xf32>) -> tensor<1x12x384x64xf32>
    %5 = migraphx.reshape(%arg2) {dims = [1, 384, 36, 64]} : (tensor<1x384x2304xf32>) -> tensor<1x384x36x64xf32>
    %6 = migraphx.transpose(%5) {permutation = [0, 2, 1, 3]} : (tensor<1x384x36x64xf32>) -> tensor<1x36x384x64xf32>
    %7 = migraphx.slice(%6) {axes = [1], ends = [24], starts = [12]} : (tensor<1x36x384x64xf32>) -> tensor<1x12x384x64xf32>
    %8 = migraphx.transpose(%7) {permutation = [0, 1, 3, 2]} : (tensor<1x12x384x64xf32>) -> tensor<1x12x64x384xf32>
    %9 = migraphx.dot(%4, %8) : (tensor<1x12x384x64xf32>, tensor<1x12x64x384xf32>) -> tensor<1x12x384x384xf32>
    %10 = migraphx.multibroadcast(%0) {out_dyn_dims = [], out_lens = [1, 12, 384, 384]} : (tensor<1xf32>) -> tensor<1x12x384x384xf32>
    %11 = migraphx.mul(%9, %10) : (tensor<1x12x384x384xf32>, tensor<1x12x384x384xf32>) -> tensor<1x12x384x384xf32>
    %12 = migraphx.add(%11, %1) : (tensor<1x12x384x384xf32>, tensor<1x12x384x384xf32>) -> tensor<1x12x384x384xf32>
    return %12 : tensor<1x12x384x384xf32>
  }
}
