// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch %s | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut resnet50_part_53 --verifier clone - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// XFAIL: *
module {
  func.func @resnet50_part_53(%arg0: tensor<2x2048x1x1xf32> {func.read_access}, %arg1: tensor<2048x1000xf32> {func.read_access}, %arg2: tensor<1x1000xf32> {func.read_access}) -> (tensor<2x1000xf32> {func.write_access}) {
    %0 = "tosa.reshape"(%arg1) {new_shape = [1, 2048, 1000]} : (tensor<2048x1000xf32>) -> tensor<1x2048x1000xf32>
    %1 = "tosa.reshape"(%arg0) {new_shape = [1, 2, 2048]} : (tensor<2x2048x1x1xf32>) -> tensor<1x2x2048xf32>
    %2 = "tosa.matmul"(%1, %0) : (tensor<1x2x2048xf32>, tensor<1x2048x1000xf32>) -> tensor<1x2x1000xf32>
    %3 = "tosa.reshape"(%2) {new_shape = [2, 1000]} : (tensor<1x2x1000xf32>) -> tensor<2x1000xf32>
    %4 = "tosa.add"(%3, %arg2) : (tensor<2x1000xf32>, tensor<1x1000xf32>) -> tensor<2x1000xf32>
    return %4 : tensor<2x1000xf32>
  }
}

