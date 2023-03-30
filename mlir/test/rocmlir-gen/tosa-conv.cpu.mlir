// RUN: rocmlir-opt -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg))" %s |\
// RUN: rocmlir-opt  -linalg-fuse-elementwise-ops -empty-tensor-to-alloc-tensor -linalg-bufferize -func-bufferize -bufferization-bufferize -buffer-results-to-out-params -finalizing-bufferize |\
// RUN: rocmlir-gen -ph -pr -rand 1 -fut test_fusion - |\
// RUN: rocmlir-opt -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:   --convert-arith-to-llvm --convert-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {
  // CHECK:  [6,     2,     0,     0,     6,     6,     6,     6,     6,     0,     0,     6,     6,     0,     0,     0]
  func.func @test_fusion(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x3x3x8xf32>) -> tensor<1x30x30x16xf32> {

    %cst = arith.constant dense<0.0> : tensor<16xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %cst) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>, tensor<16x3x3x8xf32>, tensor<16xf32>) -> tensor<1x30x30x16xf32>
    %1 = "tosa.clamp"(%0) {min_fp = 0.0 : f32, max_fp = 6.0 : f32, min_int = 0 : i64, max_int = 6 : i64} : (tensor<1x30x30x16xf32>) -> tensor<1x30x30x16xf32>
    return %1 : tensor<1x30x30x16xf32>
  }

}

