// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -print-results -rand none -fut test - | \
// RUN: rocmlir-opt -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN: --convert-math-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2

module {
// CHECK: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [4] strides = [1] data =
// CHECK-NEXT: 0.0320586,  0.236883,  0.643914,  0.0871443

  func.func @create_test_tensor() -> tensor<4xf32> {
    %0 = "tosa.const"() {value = dense<[0.0, 2.0, 3.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
      return %0 : tensor<4xf32>
  }

  func.func @softmax(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "migraphx.softmax"(%arg0) {axis = 0 : i64} : (tensor<4xf32>) -> tensor<4xf32>
     return %0 : tensor<4xf32>
  }

  func.func @test() -> tensor<4xf32> {
    %0 = call @create_test_tensor() : () -> (tensor<4xf32>)
    %1 = call @softmax(%0) : (tensor<4xf32>) -> (tensor<4xf32>)
     return %1 : tensor<4xf32>
  }
}

