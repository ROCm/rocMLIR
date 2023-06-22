// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch %s | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut bert_part_48 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK-DISABLED: RMS = {{.*}}e-08
// CHECK: [1 1 1]
module {
  func.func @bert_part_48(%arg0: tensor<1x1x384xf32> {func.read_access}, %arg1: tensor<384x384xf32> {func.read_access}, %arg2: tensor<1x384xf32> {func.read_access}) -> (tensor<1x1x384xf32> {func.write_access}) {
      %0 = "tosa.reshape"(%arg1) {new_shape = array<i64: 1, 384, 384>} : (tensor<384x384xf32>) -> tensor<1x384x384xf32>
      %1 = "tosa.matmul"(%arg0, %0) : (tensor<1x1x384xf32>, tensor<1x384x384xf32>) -> tensor<1x1x384xf32>
      %2 = "tosa.reshape"(%1) {new_shape = array<i64: 1, 384>} : (tensor<1x1x384xf32>) -> tensor<1x384xf32>
      %3 = "tosa.add"(%2, %arg2) : (tensor<1x384xf32>, tensor<1x384xf32>) -> tensor<1x384xf32>
      %4 = "tosa.tanh"(%3) : (tensor<1x384xf32>) -> tensor<1x384xf32>
      %5 = "tosa.reshape"(%4) {new_shape = array<i64: 1, 1, 384>} : (tensor<1x384xf32>) -> tensor<1x1x384xf32>
      return %5 : tensor<1x1x384xf32>
    }
}
