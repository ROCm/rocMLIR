// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch %s | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut bert_part_4 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK-DISABLED: RMS = {{.*}}e-08
// CHECK: [1 1 1]
module {
  func.func @bert_part_4(%arg0: tensor<1x12x12x12xf32> {func.read_access}, %arg1: tensor<1x12x12x32xf32> {func.read_access}) -> (tensor<1x12x12x32xf32> {func.write_access}) {
      %0 = "tosa.reshape"(%arg1) {new_shape = array<i64: 12, 12, 32>} : (tensor<1x12x12x32xf32>) -> tensor<12x12x32xf32>
      %1 = "tosa.reshape"(%arg0) {new_shape = array<i64: 12, 12, 12>} : (tensor<1x12x12x12xf32>) -> tensor<12x12x12xf32>
      %2 = "tosa.matmul"(%1, %0) : (tensor<12x12x12xf32>, tensor<12x12x32xf32>) -> tensor<12x12x32xf32>
      %3 = "tosa.reshape"(%2) {new_shape = array<i64: 1, 12, 12, 32>} : (tensor<12x12x32xf32>) -> tensor<1x12x12x32xf32>
      return %3 : tensor<1x12x12x32xf32>
    }
}
