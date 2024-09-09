// RUN: rocmlir-opt -convert-rock-to-gpu --split-input-file %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @emptykernel_module
// CHECK-NEXT: gpu.func @emptykernel(%{{.*}}: memref<?x?x?x?xf32> {llvm.noalias}) kernel
module {
  func.func @emptykernel(%arg0: memref<?x?x?x?xf32> {llvm.noalias}) attributes {kernel = 0 : i32} {
    return
  }
}

// -----

// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @emptykernel_module
// CHECK-NEXT: gpu.func @emptykernel(%{{.*}}: memref<?x?x?x?xf32> {llvm.noalias}) kernel
// CHECK-SAME: arch = "gfx90a"
module {
  func.func @emptykernel(%arg0: memref<?x?x?x?xf32> {llvm.noalias}) attributes {kernel = 0 : i32, arch = "gfx90a"} {
    return
  }
}
