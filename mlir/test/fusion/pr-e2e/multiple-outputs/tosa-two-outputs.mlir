// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch %s | rocmlir-gen -ph -print-results -fut test_mo -verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK-COUNT-2:  [1 1 1]
module {
  func.func @test_mo(%arg0: tensor<1x256x768xf32>, %arg1: tensor<1x768x768xf32>, %arg2: tensor<1x256x1xf32>, %arg3: tensor<1x256x768xf32>) -> (tensor<1x256x768xf32>, tensor<1x256x768xf32>) {
    %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 256, 768>} : (tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
    %1 = "tosa.reshape"(%arg1) {new_shape = array<i64: 1, 768, 768>} : (tensor<1x768x768xf32>) -> tensor<1x768x768xf32>
    %2 = "tosa.matmul"(%0, %1) : (tensor<1x256x768xf32>, tensor<1x768x768xf32>) -> tensor<1x256x768xf32>
    %3 = "tosa.reshape"(%2) {new_shape = array<i64: 1, 256, 768>} : (tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
    %4 = "tosa.add"(%3, %arg2) : (tensor<1x256x768xf32>, tensor<1x256x1xf32>) -> tensor<1x256x768xf32>
    %5 = "tosa.add"(%4, %arg3) : (tensor<1x256x768xf32>, tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
    %6 = "tosa.clamp"(%5) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
    return %5, %6 : tensor<1x256x768xf32>, tensor<1x256x768xf32>
  }
}
