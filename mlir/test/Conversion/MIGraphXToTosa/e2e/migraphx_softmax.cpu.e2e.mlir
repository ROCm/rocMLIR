// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-gen -ph -print-results -rand none -fut test - | \
// RUN: rocmlir-opt -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN: --convert-math-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2

module {
// CHECK: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [4] strides = [1] data =
// CHECK-NEXT: 0.0320586,  0.236883,  0.643914,  0.0871443

  func.func @create_test_tensor() -> !migraphx.shaped<4xf32, 1> {
    %0 = migraphx.literal (dense<[0.0, 2.0, 3.0, 1.0]> : tensor<4xf32>) : <4xf32, 1>
    return %0 : !migraphx.shaped<4xf32, 1>
  }

  func.func @softmax(%arg0: !migraphx.shaped<4xf32, 1>) -> !migraphx.shaped<4xf32, 1> {
    %0 = migraphx.softmax %arg0 {axis = 0 : i64} : <4xf32, 1> -> <4xf32, 1>
     return %0 : !migraphx.shaped<4xf32, 1>
  }

  func.func @test() -> !migraphx.shaped<4xf32, 1> {
    %0 = call @create_test_tensor() : () -> (!migraphx.shaped<4xf32, 1>)
    %1 = call @softmax(%0) : (!migraphx.shaped<4xf32, 1>) -> (!migraphx.shaped<4xf32, 1>)
     return %1 : !migraphx.shaped<4xf32, 1>
  }
}

