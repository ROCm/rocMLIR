// RUN: rocmlir-opt -convert-rock-to-gpu %s | FileCheck %s

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

    // CHECK-NOT: llvm.intr.assume
    %idx = arith.muli %bid, %tid : index

    %val = memref.load %arg0[%idx] : memref<?xf32>

    memref.store %val, %arg1[%idx] : memref<?xf32>
    return
  }

  // CHECK-LABEL: func @launch_dims
  func.func @launch_dims(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {kernel = 0 : i32, block_size = 64 : i32, grid_size = 512 : i32} {
    // CHECK-DAG: %[[c64:.*]] = arith.constant 64 : index
    // CHECK-DAG: %[[c512:.*]] = arith.constant 512 : index
    // CHECK: %[[bid:.*]] = gpu.block_id x
    // CHECK: %[[cmpGroup:.*]] = arith.cmpi ult, %[[bid]], %[[c512]]
    // CHECK: "llvm.intr.assume"(%[[cmpGroup]]) : (i1) -> ()
    %bid = rock.workgroup_id : index

    // CHECK: %[[tid:.*]] = gpu.thread_id x
    // CHECK: %[[cmpBlock:.*]] = arith.cmpi ult, %[[tid]], %[[c64]]
    // CHECK: "llvm.intr.assume"(%[[cmpBlock]]) : (i1) -> ()
    %tid = rock.workitem_id : index

   %idx = arith.muli %bid, %tid : index

    %val = memref.load %arg0[%idx] : memref<?xf32>

    memref.store %val, %arg1[%idx] : memref<?xf32>
    return
  }
}
