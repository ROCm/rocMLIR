// RUN: python3 %S/Inputs/resnet50-test.py %S/Inputs/590px-Red_Smooth_Saluki.jpg \
//    | tf-opt -tensor-bufferize --tf-to-tosa-pipeline \
//    | mlir-opt --tosa-partition --tosa-to-linalg-on-tensors --tosa-to-standard \
//      --linalg-detensorize -tensor-constant-bufferize -std-bufferize -linalg-bufferize \
//      -tensor-bufferize -func-bufferize -finalizing-bufferize --convert-linalg-to-loops \
//      --tosa-to-standard -lower-affine -convert-linalg-to-llvm --convert-scf-to-std \
//      --convert-math-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts \
//    | mlir-cpu-runner -e main -entry-point-result=void \
//      -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
//      -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
//    | FileCheck %s

// CHECK:  Expected result is [176]
// CHECK:  Unranked Memref
// CHECK:  [176]
