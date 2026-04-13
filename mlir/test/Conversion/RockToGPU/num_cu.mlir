// RUN: rocmlir-opt -convert-rock-to-gpu -split-input-file %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @misckernel_module
// CHECK-NEXT: gpu.func @misckernel(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>) 
// CHECK-SAME: workgroup(%arg2 : memref<64xf32, #gpu.address_space<workgroup>> {llvm.align = 64 : i64})
// CHECK-SAME: kernel
// CHECK-SAME: block_size = 128 : i32
// CHECK-SAME: grid_size = 256 : i32
// CHECK-SAME: known_block_size = array<i32: 128, 1, 1>
// CHECK-SAME: known_grid_size = array<i32: 256, 1, 1>
// CHECK-SAME: rocdl.unsafe_fp_atomics = true
// CHECK-SAME: rocdl.waves_per_eu = 2 : i32
// CHECK-SAME: rock.arch = "amdgcn-amd-amdhsa:gfx1100"
// CHECK-SAME: rock.num_cu = 96 : i64
// CHECK-SAME: rock.shared_buffer_size = 256 : i32
module {
  func.func @misckernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {block_size = 128 : i32, rock.enable_splitk_for_tuning, features = #rock<GemmFeatures wmma|dot|atomic_add|atomic_fmax_f32>, grid_size = 256 : i32, rock.kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100", rock.num_cu = 96 : i64} {
    %lds = rock.alloc() : memref<64xf32, #gpu.address_space<workgroup>>
    
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
    %val_lds = memref.load %lds[%idx] : memref<64xf32, #gpu.address_space<workgroup>>

    memref.store %val, %arg1[%idx] : memref<?xf32>
    memref.store %val_lds, %arg1[%idx] : memref<?xf32>
    return
  }
}
