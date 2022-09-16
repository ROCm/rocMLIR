// RUN: mlir-rock-driver -host-pipeline=highlevel %s | rock-gen -rand=none -ph -pr -fut test_fusion - | rock-opt -convert-linalg-to-loops -lower-affine -convert-scf-to-cf | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {

  // CHECK:  [2,     2,     2,     2,     2,     2,     2,     2]
  func.func @test_fusion(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x32x32x8xf32>, tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
    return %0 : tensor<1x32x32x8xf32>
  }

}

