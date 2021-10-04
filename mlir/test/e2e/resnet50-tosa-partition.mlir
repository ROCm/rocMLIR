// RUN:   python3 %S/Inputs/resnet50-test.py %S/Inputs/590px-Red_Smooth_Saluki.jpg \
// RUN: | /tensorflow/bazel-bin/tensorflow/compiler/mlir/tf-opt -tensor-bufferize --tf-to-tosa-pipeline \
// RUN: | mlir-opt --tosa-partition-pipeline --tosa-to-linalg-on-tensors --tosa-to-standard \
// RUN:   --linalg-detensorize -tensor-constant-bufferize -std-bufferize -linalg-bufferize \
// RUN:   -tensor-bufferize -func-bufferize -finalizing-bufferize --convert-linalg-to-loops \
// RUN:   --tosa-to-standard -lower-affine -convert-linalg-to-llvm --convert-scf-to-std \
// RUN:   --convert-math-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

// python will print "Expected result is [176]" but on stderr and early in
//   pipeline, so we can't check for it here.
// CHECK:  Unranked Memref
// CHECK:  [176]
