// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=emptykernel" %s | FileCheck %s
// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=emptykernel gpu-module-name=foomodule" %s | FileCheck %s --check-prefix=MODULE

// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @miopen_kernel_module
// CHECK-NEXT: gpu.func @emptykernel(%{{.*}}: memref<?x?x?x?xf32>) kernel

// MODULE: module attributes {gpu.container_module}
// MODULE-NEXT: gpu.module @foomodule
// MODULE-NEXT: gpu.func @emptykernel(%{{.*}}: memref<?x?x?x?xf32>) kernel
module {
  func @emptykernel(%arg0: memref<?x?x?x?xf32>) {
    return
  }
}
