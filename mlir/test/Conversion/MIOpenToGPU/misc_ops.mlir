// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=misckernel" %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @miopen_kernel_module
// CHECK-NEXT: gpu.func @misckernel(%{{.*}}: memref<?x?x?x?xf32>) kernel
module {
  func @misckernel(%arg0: memref<?x?x?x?xf32>) {
    // CHECK: gpu.barrier
    miopen.workgroup_barrier

    // CHECK: %{{.*}} = "gpu.block_id"() {dimension = "x"} : () -> index
    %bid = miopen.workgroup_id : index

    // CHECK: %{{.*}} = "gpu.thread_id"() {dimension = "x"} : () -> index
    %tid = miopen.workitem_id : index
    return
  }
}
