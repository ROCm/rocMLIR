// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=existing_kernel gpu-module-name=existing_module" %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @existing_module
// CHECK-NEXT: gpu.func @existing_kernel(%{{.*}}: memref<?x?x?x?xf32>) kernel
module {
  gpu.module @existing_module {
    gpu.func @existing_kernel(%arg0: memref<?x?x?x?xf32>) {
      gpu.return
    }
  }
}
