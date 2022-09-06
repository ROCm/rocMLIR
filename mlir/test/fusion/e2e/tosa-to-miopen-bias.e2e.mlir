// RUN: mlir-miopen-driver -host-pipeline highlevel %s | miopen-gen -ph -print-results -rand none - | mlir-miopen-driver -c  | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext -entry-point-result=void | FileCheck %s
// CHECK: Unranked Memref base
func.func @test_fusion(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x3x3x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x30x30x16xf32> attributes {kernel} {
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x32x32x8xf32>, tensor<16x3x3x8xf32>, tensor<16xf32>) -> tensor<1x30x30x16xf32>

  return %0 : tensor<1x30x30x16xf32>
}

// -----

