// RUN: rock-opt -convert-rock-to-gpu %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK: gpu.module @emptykernel_module
// CHECK-NEXT: gpu.func @emptykernel(%{{.*}}: memref<?x?x?x?xf32>) kernel
module {
  gpu.module @existing_module {
  }
  func.func @emptykernel(%arg0: memref<?x?x?x?xf32>) attributes {kernel = 0 : i32} {
    return
  }
}
