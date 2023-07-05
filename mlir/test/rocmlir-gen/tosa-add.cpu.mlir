// RUN: rocmlir-driver -host-pipeline=highlevel %s | rocmlir-gen -rand=none -ph -pr -fut test_fusion -\
// RUN: | rocmlir-opt -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:   --convert-arith-to-llvm --expand-strided-metadata --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {

  // CHECK:  [2,     2,     2,     2,     2,     2,     2,     2]
  func.func @test_fusion(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x32x32x8xf32>, tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
    return %0 : tensor<1x32x32x8xf32>
  }

}

