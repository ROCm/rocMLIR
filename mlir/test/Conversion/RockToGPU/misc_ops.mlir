// RUN: rock-opt -convert-rock-to-gpu %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @misckernel_module
// CHECK-NEXT: gpu.func @misckernel(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>) kernel
module {
  func.func @misckernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {kernel = 0 : i32} {
    // CHECK: gpu.barrier
    rock.workgroup_barrier

    // CHECK: gpu.lds_barrier
    rock.lds_barrier

    // CHECK: %{{.*}} = gpu.block_id x
    %bid = rock.workgroup_id : index

    // CHECK: %{{.*}} = gpu.thread_id x
    %tid = rock.workitem_id : index

    %idx = arith.muli %bid, %tid : index

    %val = memref.load %arg0[%idx] : memref<?xf32>

    memref.store %val, %arg1[%idx] : memref<?xf32>
    return
  }
}
