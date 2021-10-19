// RUN: miopen-opt -convert-miopen-to-gpu %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @existing_module
// CHECK-NEXT: gpu.func @existing_kernel(%{{.*}}: memref<?x?x?x?xf32>) kernel
module {
  gpu.module @existing_module {
    gpu.func @existing_kernel(%arg0: memref<?x?x?x?xf32>) kernel {
      gpu.return
    }
  }
}
