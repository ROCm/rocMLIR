// RUN: rocmlir-opt %s --rock-block-pingpong | FileCheck %s
//
// Test that block ping-pong pass inserts scheduling hints:
// - sched_barrier around existing LDS barriers (cluster boundaries)
// - setprio around MFMA operations (when present)
//
// Note: Full phase shift with conditional barriers is not implemented due to
// AMD's s_barrier counting barrier semantics. With multiple barriers per loop
// iteration, phase shift would cause deadlocks.

// CHECK-LABEL: func.func @kernel_8_waves_double_buffered
func.func @kernel_8_waves_double_buffered() attributes {
  arch = "amdgcn-amd-amdhsa:gfx90a",
  block_size = 512 : i32,
  grid_size = 1 : i32,
  rock.use_block_pingpong,
  "rock.double_buffered"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %rawLds = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
  %lds = memref.view %rawLds[%c0][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<16xi8, #gpu.address_space<workgroup>>

  // 8-wave mode with double-buffering: scheduling hints only (no phase shift)
  // CHECK-NOT: rock.cond_barrier
  // CHECK: scf.for
  scf.for %iv = %c0 to %c4 step %c1 {
    // Inside loop: sched_barrier BEFORE and AFTER existing lds_barrier
    // CHECK: amdgpu.sched_barrier allow = <none>
    // CHECK-NEXT: rock.lds_barrier
    // CHECK-NEXT: amdgpu.sched_barrier allow = <none>
    rock.lds_barrier
    %v = memref.load %lds[%iv] : memref<16xi8, #gpu.address_space<workgroup>>
    memref.store %v, %lds[%iv] : memref<16xi8, #gpu.address_space<workgroup>>
  }
  // No conditional barriers
  // CHECK-NOT: rock.cond_barrier
  // CHECK: return
  return
}

// Test: 8-wave mode without double-buffering (single-buffered LDS)
// Should only apply cluster boundaries, no phase shift
// CHECK-LABEL: func.func @kernel_8_waves_single_buffered
func.func @kernel_8_waves_single_buffered() attributes {
  arch = "amdgcn-amd-amdhsa:gfx90a",
  block_size = 512 : i32,
  grid_size = 1 : i32,
  rock.use_block_pingpong
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %rawLds = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
  %lds = memref.view %rawLds[%c0][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<16xi8, #gpu.address_space<workgroup>>

  // 8-wave mode without double-buffering: no phase shift, only cluster boundaries
  // CHECK-NOT: rock.cond_barrier
  // CHECK: scf.for
  scf.for %iv = %c0 to %c4 step %c1 {
    // Inside loop: sched_barrier BEFORE and AFTER existing lds_barrier
    // CHECK: amdgpu.sched_barrier allow = <none>
    // CHECK-NEXT: rock.lds_barrier
    // CHECK-NEXT: amdgpu.sched_barrier allow = <none>
    rock.lds_barrier
    %v = memref.load %lds[%iv] : memref<16xi8, #gpu.address_space<workgroup>>
    memref.store %v, %lds[%iv] : memref<16xi8, #gpu.address_space<workgroup>>
  }
  // No cond_barrier after loop (phase shift disabled)
  // CHECK-NOT: rock.cond_barrier
  // CHECK: return
  return
}

// Test: 4 waves with sufficient grid size should apply cluster-only mode
// CHECK-LABEL: func.func @kernel_4_waves_cluster_only
func.func @kernel_4_waves_cluster_only() attributes {
  arch = "amdgcn-amd-amdhsa:gfx90a",
  block_size = 256 : i32,
  grid_size = 220 : i32,
  rock.use_block_pingpong
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %rawLds = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
  %lds = memref.view %rawLds[%c0][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<16xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.cond_barrier
  // CHECK: scf.for
  scf.for %iv = %c0 to %c4 step %c1 {
    // Should have sched_barrier BEFORE and AFTER existing lds_barrier
    // CHECK: amdgpu.sched_barrier allow = <none>
    // CHECK-NEXT: rock.lds_barrier
    // CHECK-NEXT: amdgpu.sched_barrier allow = <none>
    rock.lds_barrier
    %v = memref.load %lds[%iv] : memref<16xi8, #gpu.address_space<workgroup>>
    memref.store %v, %lds[%iv] : memref<16xi8, #gpu.address_space<workgroup>>
  }
  // No cond_barrier after loop in 4-wave mode
  // CHECK-NOT: rock.cond_barrier
  // CHECK: return
  return
}

// Test: 4 waves with assume_2_wg_per_cu should apply cluster-only mode
// CHECK-LABEL: func.func @kernel_4_waves_assume_2wg
func.func @kernel_4_waves_assume_2wg() attributes {
  arch = "amdgcn-amd-amdhsa:gfx90a",
  block_size = 256 : i32,
  grid_size = 1 : i32,
  rock.use_block_pingpong,
  rock.assume_2_wg_per_cu
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %rawLds = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
  %lds = memref.view %rawLds[%c0][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<16xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.cond_barrier
  // CHECK: scf.for
  scf.for %iv = %c0 to %c4 step %c1 {
    // CHECK: amdgpu.sched_barrier allow = <none>
    // CHECK-NEXT: rock.lds_barrier
    // CHECK-NEXT: amdgpu.sched_barrier allow = <none>
    rock.lds_barrier
    %v = memref.load %lds[%iv] : memref<16xi8, #gpu.address_space<workgroup>>
    memref.store %v, %lds[%iv] : memref<16xi8, #gpu.address_space<workgroup>>
  }
  // CHECK: return
  return
}

// Test: 4 waves with insufficient grid size should skip
// CHECK-LABEL: func.func @kernel_4_waves_insufficient_grid
func.func @kernel_4_waves_insufficient_grid() attributes {
  arch = "amdgcn-amd-amdhsa:gfx90a",
  block_size = 256 : i32,
  grid_size = 50 : i32,
  rock.use_block_pingpong
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %rawLds = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
  %lds = memref.view %rawLds[%c0][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<16xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.cond_barrier
  // CHECK-NOT: amdgpu.sched_barrier
  // CHECK: scf.for
  scf.for %iv = %c0 to %c4 step %c1 {
    rock.lds_barrier
    %v = memref.load %lds[%iv] : memref<16xi8, #gpu.address_space<workgroup>>
    memref.store %v, %lds[%iv] : memref<16xi8, #gpu.address_space<workgroup>>
  }
  // CHECK: return
  return
}

// Test: No rock.use_block_pingpong attr - should NOT apply any transformation
// CHECK-LABEL: func.func @kernel_no_pingpong_attr
// CHECK-NOT: rock.cond_barrier
// CHECK-NOT: amdgpu.sched_barrier
// CHECK: scf.for
func.func @kernel_no_pingpong_attr() attributes {
  arch = "amdgcn-amd-amdhsa:gfx90a",
  block_size = 512 : i32,
  grid_size = 1 : i32
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %rawLds = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
  %lds = memref.view %rawLds[%c0][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<16xi8, #gpu.address_space<workgroup>>
  scf.for %iv = %c0 to %c4 step %c1 {
    rock.lds_barrier
    %v = memref.load %lds[%iv] : memref<16xi8, #gpu.address_space<workgroup>>
    memref.store %v, %lds[%iv] : memref<16xi8, #gpu.address_space<workgroup>>
  }
  // CHECK: return
  return
}
