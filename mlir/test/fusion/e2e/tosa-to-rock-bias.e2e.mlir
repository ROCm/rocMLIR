// RUN: rocmlir-driver -host-pipeline highlevel %s | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext -entry-point-result=void | FileCheck %s
// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch %s | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut test_fusion --verifier clone - | rocmlir-driver -host-pipeline xmodel,runner -kernel-pipeline full -targets %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
// CHECK: Unranked Memref base
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base
func.func @test_fusion(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x3x3x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x30x30x16xf32> attributes {kernel, arch = ""} {
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>, tensor<16x3x3x8xf32>, tensor<16xf32>) -> tensor<1x30x30x16xf32>

  return %0 : tensor<1x30x30x16xf32>
}

// -----
