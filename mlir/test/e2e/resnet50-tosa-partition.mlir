// RUN:   python3 %S/Inputs/resnet50-test.py %S/Inputs/590px-Red_Smooth_Saluki.jpg \
// RUN: | mlir-opt --tosa-partition-pipeline --tosa-to-linalg-on-tensors --tosa-to-standard \
// RUN:   --linalg-detensorize -tensor-constant-bufferize -std-bufferize -linalg-bufferize \
// RUN:   -tensor-bufferize -func-bufferize -finalizing-bufferize --convert-linalg-to-loops \
// RUN:   --tosa-to-standard -lower-affine -convert-linalg-to-llvm --convert-scf-to-std \
// RUN:   --convert-math-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/../lib/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%mlir_runner_utils_dir/../lib/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

// CHECK:  Unranked Memref
// CHECK:  [176]
