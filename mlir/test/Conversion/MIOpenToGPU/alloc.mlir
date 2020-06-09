// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=allockernel" %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @miopen_kernel_module
// CHECK-NEXT: gpu.func @allockernel(%{{.*}}: memref<?x?x?x?xf32>) workgroup(%{{.*}}: memref<16xf32, 3>) private(%{{.*}}: memref<16xf32, 5>) kernel
module {
  func @allockernel(%arg0: memref<?x?x?x?xf32>) {
    %buffer_lds = miopen.alloc() : memref<16xf32, 3>
    %buffer_vgpr = miopen.alloc() : memref<16xf32, 5>
    return
  }
}
