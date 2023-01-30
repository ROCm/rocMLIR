// RUN: rocmlir-opt --tosa-partition="anchor-ops=tosa.reduce_sum" --xmodel-async-graph --xmodel-target-kernels="targets=%arch" %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut test_basic --verifier clone - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// cat %s | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base

func.func @test_basic(%arg0: tensor<200x100xf32>) -> tensor<200x1xf32> {
  %1 = "tosa.reduce_sum"(%arg0) {axis = 1 : i64} : (tensor<200x100xf32>) -> tensor<200x1xf32>
  return %1 : tensor<200x1xf32>
}
