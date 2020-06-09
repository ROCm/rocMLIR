// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=emptykernel" %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @miopen_kernel_module
// CHECK-NEXT: gpu.func @emptykernel(%{{.*}}: memref<?x?x?x?xf32>) kernel
module {
  func @emptykernel(%arg0: memref<?x?x?x?xf32>) {
    return
  }
}
