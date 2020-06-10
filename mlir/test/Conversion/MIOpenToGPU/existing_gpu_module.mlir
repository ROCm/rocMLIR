// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=emptykernel gpu-module-name=existing_module" %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @existing_module
// CHECK-NEXT: gpu.func @emptykernel(%{{.*}}: memref<?x?x?x?xf32>) kernel
module {
  gpu.module @existing_module {
  }
  func @emptykernel(%arg0: memref<?x?x?x?xf32>) {
    return
  }
}
