// ALLOW_RETRIES: 2
// MLIR#764: rock-to-gpu bug when global constant not cloned
// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch %s | FileCheck %s
// RUN-DISABLE: rocmlir-driver -host-pipeline partition,highlevel -targets %arch %s | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut bert_part_49 --verifier clone - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext,%linalg_test_lib_dir/%prefix_mlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%prefix_mlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK-DISABLED: RMS = {{.*}}e-08
// COM: CHECK: [1 1 1]
// CHECK: bert_part_49__part_0
module {
  func.func @bert_part_49(%arg0: tensor<1x1x384xf32> {func.read_access}, %arg1: tensor<384x2xf32> {func.read_access}) -> (tensor<1x2xf32> {func.write_access}) {
      %0 = "tosa.reshape"(%arg1) {new_shape = array<i64: 1, 384, 2>} : (tensor<384x2xf32>) -> tensor<1x384x2xf32>
      %1 = "tosa.matmul"(%arg0, %0) : (tensor<1x1x384xf32>, tensor<1x384x2xf32>) -> tensor<1x1x2xf32>
      %2 = "tosa.reshape"(%1) {new_shape = array<i64: 1, 2>} : (tensor<1x1x2xf32>) -> tensor<1x2xf32>
      %3 = "tosa.const"() {value = dense<[[-0.00115577725, 0.00115577038]]> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
      %4 = "tosa.add"(%2, %3) : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
      return %4 : tensor<1x2xf32>
  }
}
