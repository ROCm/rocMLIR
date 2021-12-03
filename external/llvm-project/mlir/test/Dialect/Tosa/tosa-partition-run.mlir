// RUN: mlir-opt %s --tosa-to-linalg --tosa-to-standard --linalg-detensorize \
// RUN:   -tensor-constant-bufferize -std-bufferize -linalg-bufferize -tensor-bufferize \
// RUN:   -func-bufferize -finalizing-bufferize --convert-linalg-to-loops \
// RUN:   --tosa-to-standard -lower-affine -convert-linalg-to-llvm --convert-scf-to-std \
// RUN:   --convert-math-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: > %t1
//
// RUN  cat %t1 | FileCheck %s
// CHECK:  Unranked Memref
// CHECK-SAME:  sizes = [1, 10, 10, 2] strides = [200, 20, 2, 1]
// CHECK-NEXT:  [[[[0.458882,     0.344086],
// CHECK-NEXT:     [0.336863,     0.469958],
// CHECK-NEXT:     [0.214844,     0.595831],
// CHECK-NEXT:     [0.0928252,     0.721703],
//
// RUN: mlir-opt %s --tosa-partition-pipeline --tosa-to-linalg --tosa-to-standard --linalg-detensorize \
// RUN:   -tensor-constant-bufferize -std-bufferize -linalg-bufferize -tensor-bufferize \
// RUN:   -func-bufferize -finalizing-bufferize --convert-linalg-to-loops \
// RUN:   --tosa-to-standard -lower-affine -convert-linalg-to-llvm --convert-scf-to-std \
// RUN:   --convert-math-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: > %t2
//
// RUN: diff --ignore-matching-lines='Unranked Memref' %t1 %t2

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 808 : i32}}  {
  func private @print_memref_f32(memref<*xf32>)
  func private @printNewline()
  func @main() {
    %0 = "tosa.const"() {value = dense<[[[[-1.747810e-01], [0.356973231], [-0.166753888]], [[-0.298198819], [0.110798746], [-0.314905882]], [[-0.267817706], [0.318314373], [0.329790294]]], [[[-0.236702681], [-0.0709462166], [-0.192342982]], [[0.138438225], [-0.217499733], [0.0627906919]], [[0.0631466805], [-2.780110e-01], [0.357007563]]]]> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
    %1 = "tosa.const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %2 = "tosa.const"() {value = dense<[[[[1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [-1.000000e+00], [-2.000000e+00], [-3.000000e+00], [-4.000000e+00], [-5.000000e+00]], [[1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [-1.000000e+00], [-2.000000e+00], [-3.000000e+00], [-4.000000e+00], [-5.000000e+00]], [[1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [-1.000000e+00], [-2.000000e+00], [-3.000000e+00], [-4.000000e+00], [-5.000000e+00]], [[1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [-1.000000e+00], [-2.000000e+00], [-3.000000e+00], [-4.000000e+00], [-5.000000e+00]], [[1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [-1.000000e+00], [-2.000000e+00], [-3.000000e+00], [-4.000000e+00], [-5.000000e+00]], [[1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [-1.000000e+00], [-2.000000e+00], [-3.000000e+00], [-4.000000e+00], [-5.000000e+00]], [[1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [-1.000000e+00], [-2.000000e+00], [-3.000000e+00], [-4.000000e+00], [-5.000000e+00]], [[1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [-1.000000e+00], [-2.000000e+00], [-3.000000e+00], [-4.000000e+00], [-5.000000e+00]], [[1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [-1.000000e+00], [-2.000000e+00], [-3.000000e+00], [-4.000000e+00], [-5.000000e+00]], [[1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00], [5.000000e+00], [-1.000000e+00], [-2.000000e+00], [-3.000000e+00], [-4.000000e+00], [-5.000000e+00]]]]> : tensor<1x10x10x1xf32>} : () -> tensor<1x10x10x1xf32>
    %4 = "tosa.conv2d"(%2, %0, %1) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x10x10x1xf32>, tensor<2x3x3x1xf32>, tensor<2xf32>) -> tensor<1x10x10x2xf32>
    %5 = "tosa.clamp"(%4) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x10x10x2xf32>) -> tensor<1x10x10x2xf32>
    %8 = bufferization.to_memref %5 : memref<1x10x10x2xf32>
    %9 = memref.cast %8 : memref<1x10x10x2xf32> to memref<*xf32>
    call @print_memref_f32(%9) : (memref<*xf32>) -> ()
    call @printNewline() : () -> ()
    return
  }
}
