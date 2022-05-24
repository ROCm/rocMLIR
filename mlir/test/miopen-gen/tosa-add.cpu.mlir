// RUN: mlir-miopen-driver -host-pipeline=highlevel %s | miopen-gen -rand=none -ph -pvr -fut test_fusion - | miopen-opt -convert-linalg-to-loops -lower-affine -convert-scf-to-cf | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {

  // CHECK:  [2,     2,     2,     2,     2,     2,     2,     2]
  func @test_fusion(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x32x32x8xf32>, tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
    return %0 : tensor<1x32x32x8xf32>
  }

}

