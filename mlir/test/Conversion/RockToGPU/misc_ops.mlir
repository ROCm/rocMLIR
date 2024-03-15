// RUN: rocmlir-opt -convert-rock-to-gpu -split-input-file %s | FileCheck %s


// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @misckernel_module
// CHECK-NEXT: gpu.func @misckernel(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>) kernel
// CHECK-SAME: block_size = 64 : i32
// CHECK-SAME: gpu.known_block_size = array<i32: 64, 1, 1>
// CHECK-SAME: gpu.known_grid_size = array<i32: 900, 1, 1>
// CHECK-SAME: grid_size = 900 : i32
module {
  func.func @misckernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>)
      attributes {kernel = 0 : i32, block_size = 64 : i32, grid_size = 900 : i32} {
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

// -----

module {
  func.func @misckernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>)
      attributes {kernel = 0 : i32, block_size = 64 : i32, grid_size = 900 : i32, reverse_grid} {
    // CHECK: %[[BLOCKID:.+]] = gpu.block_id x
    // CHECK: arith.subi %c899, %[[BLOCKID]]
    %bid = rock.workgroup_id : index
    %val = memref.load %arg0[%bid] : memref<?xf32>
    memref.store %val, %arg1[%bid] : memref<?xf32>
    return
  }
}


