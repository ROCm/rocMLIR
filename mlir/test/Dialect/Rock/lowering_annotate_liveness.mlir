// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-annotate-liveness | FileCheck %s

#wg = #gpu.address_space<workgroup>

// CHECK-LABEL: func.func @rock_reuse_two
func.func @rock_reuse_two() attributes{arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  // CHECK: %[[ALLOC1:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  %0 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[ALLOC2:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  %1 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[ALLOC3:.*]] = rock.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
  %2 = rock.alloc() : memref<2048xi8, #wg>
  %c0 = arith.constant 0 : index
  %view0 = memref.view %0[%c0][] : memref<1024xi8, #wg> to memref<256xf32, #wg>
  %view1 = memref.view %1[%c0][] : memref<1024xi8, #wg> to memref<256xf32, #wg>
  %view2 = memref.view %2[%c0][] : memref<2048xi8, #wg> to memref<512xf32, #wg>
  %cst_0 = arith.constant 0.000000e+00 : f32

  // CHECK: rock.live_in %[[ALLOC1]]
  rock.in_bounds_store %cst_0 -> %view0[%c0] : f32 -> memref<256xf32, #wg>, index
  // CHECK: rock.live_in %[[ALLOC2]]
  rock.in_bounds_store %cst_0 -> %view1[%c0] : f32 -> memref<256xf32, #wg>, index
  // CHECK: rock.live_out %[[ALLOC1]]
  %load0 = rock.in_bounds_load %view0[%c0] : memref<256xf32, #wg>, index -> f32
  // CHECK: rock.live_out %[[ALLOC2]]
  %load1 = rock.in_bounds_load %view1[%c0] : memref<256xf32, #wg>, index -> f32

  // CHECK: rock.live_in %[[ALLOC3]]
  rock.in_bounds_store %cst_0 -> %view2[%c0] : f32 -> memref<512xf32, #wg>, index
  // CHECK: rock.live_out %[[ALLOC3]]
  %load2 = rock.in_bounds_load %view2[%c0] : memref<512xf32, #wg>, index -> f32

  // CHECK-NOT: rock.live_in
  // CHECK-NOT: rock.live_out
  
  return
}

// CHECK-LABEL: func.func @rock_multiple_stores_and_loads
func.func @rock_multiple_stores_and_loads() attributes{arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  // CHECK: %[[ALLOC1:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  %0 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[ALLOC2:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  %1 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[ALLOC3:.*]] = rock.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
  %2 = rock.alloc() : memref<2048xi8, #wg>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %view0 = memref.view %0[%c0][] : memref<1024xi8, #wg> to memref<256xf32, #wg>
  %view1 = memref.view %1[%c0][] : memref<1024xi8, #wg> to memref<256xf32, #wg>
  %view2 = memref.view %2[%c0][] : memref<2048xi8, #wg> to memref<512xf32, #wg>
  %cst_0 = arith.constant 0.000000e+00 : f32

  // CHECK: rock.live_in %[[ALLOC1]]
  rock.in_bounds_store %cst_0 -> %view0[%c0] : f32 -> memref<256xf32, #wg>, index
  rock.in_bounds_store %cst_0 -> %view0[%c0] : f32 -> memref<256xf32, #wg>, index
  rock.in_bounds_store %cst_0 -> %view0[%c1] : f32 -> memref<256xf32, #wg>, index
  rock.in_bounds_store %cst_0 -> %view0[%c2] : f32 -> memref<256xf32, #wg>, index
  // CHECK: rock.live_in %[[ALLOC2]]
  rock.in_bounds_store %cst_0 -> %view1[%c0] : f32 -> memref<256xf32, #wg>, index
  rock.in_bounds_store %cst_0 -> %view1[%c0] : f32 -> memref<256xf32, #wg>, index
  rock.in_bounds_store %cst_0 -> %view1[%c1] : f32 -> memref<256xf32, #wg>, index
  rock.in_bounds_store %cst_0 -> %view1[%c2] : f32 -> memref<256xf32, #wg>, index
  // CHECK: rock.live_out %[[ALLOC1]]
  %load0_0 = rock.in_bounds_load %view0[%c0] : memref<256xf32, #wg>, index -> f32
  %load0_0_repeat = rock.in_bounds_load %view0[%c0] : memref<256xf32, #wg>, index -> f32
  %load0_1 = rock.in_bounds_load %view0[%c1] : memref<256xf32, #wg>, index -> f32
  %load0_2 = rock.in_bounds_load %view0[%c2] : memref<256xf32, #wg>, index -> f32
  // CHECK: rock.live_out %[[ALLOC2]]
  %load1_0 = rock.in_bounds_load %view1[%c0] : memref<256xf32, #wg>, index -> f32
  %load1_0_repeat = rock.in_bounds_load %view1[%c0] : memref<256xf32, #wg>, index -> f32
  %load1_1 = rock.in_bounds_load %view1[%c1] : memref<256xf32, #wg>, index -> f32
  %load1_2 = rock.in_bounds_load %view1[%c2] : memref<256xf32, #wg>, index -> f32

  // CHECK: rock.live_in %[[ALLOC3]]
  rock.in_bounds_store %cst_0 -> %view2[%c0] : f32 -> memref<512xf32, #wg>, index
  rock.in_bounds_store %cst_0 -> %view2[%c0] : f32 -> memref<512xf32, #wg>, index
  rock.in_bounds_store %cst_0 -> %view2[%c1] : f32 -> memref<512xf32, #wg>, index
  rock.in_bounds_store %cst_0 -> %view2[%c2] : f32 -> memref<512xf32, #wg>, index
  // CHECK: rock.live_out %[[ALLOC3]]
  %load2_0 = rock.in_bounds_load %view2[%c0] : memref<512xf32, #wg>, index -> f32
  %load2_0_repeat = rock.in_bounds_load %view2[%c0] : memref<512xf32, #wg>, index -> f32
  %load2_1 = rock.in_bounds_load %view2[%c1] : memref<512xf32, #wg>, index -> f32
  %load2_2 = rock.in_bounds_load %view2[%c2] : memref<512xf32, #wg>, index -> f32

  // CHECK-NOT: rock.live_in
  // CHECK-NOT: rock.live_out

  return
}

// CHECK-LABEL: func.func @rock_multiple_liveness_ranges
func.func @rock_multiple_liveness_ranges() attributes{arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  // CHECK: %[[ALLOC1:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  %0 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[ALLOC2:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  %1 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[ALLOC3:.*]] = rock.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
  %2 = rock.alloc() : memref<2048xi8, #wg>
  %c0 = arith.constant 0 : index
  %view0 = memref.view %0[%c0][] : memref<1024xi8, #wg> to memref<256xf32, #wg>
  %view1 = memref.view %1[%c0][] : memref<1024xi8, #wg> to memref<256xf32, #wg>
  %view2 = memref.view %2[%c0][] : memref<2048xi8, #wg> to memref<512xf32, #wg>
  %cst_0 = arith.constant 0.000000e+00 : f32

  // CHECK: rock.live_in %[[ALLOC1]]
  rock.in_bounds_store %cst_0 -> %view0[%c0] : f32 -> memref<256xf32, #wg>, index
  // CHECK: rock.live_in %[[ALLOC2]]
  rock.in_bounds_store %cst_0 -> %view1[%c0] : f32 -> memref<256xf32, #wg>, index
  // CHECK: rock.live_out %[[ALLOC1]]
  %load0_0 = rock.in_bounds_load %view0[%c0] : memref<256xf32, #wg>, index -> f32
  // CHECK: rock.live_out %[[ALLOC2]]
  %load1_0 = rock.in_bounds_load %view1[%c0] : memref<256xf32, #wg>, index -> f32
  
  // CHECK: rock.live_in %[[ALLOC3]]
  rock.in_bounds_store %cst_0 -> %view2[%c0] : f32 -> memref<512xf32, #wg>, index
  // CHECK: rock.live_out %[[ALLOC3]]
  %load2_0 = rock.in_bounds_load %view2[%c0] : memref<512xf32, #wg>, index -> f32

  // repeat the same
  
  // CHECK: rock.live_in %[[ALLOC1]]
  rock.in_bounds_store %cst_0 -> %view0[%c0] : f32 -> memref<256xf32, #wg>, index
  // CHECK: rock.live_in %[[ALLOC2]]
  rock.in_bounds_store %cst_0 -> %view1[%c0] : f32 -> memref<256xf32, #wg>, index
  // CHECK: rock.live_out %[[ALLOC1]]
  %load0_1 = rock.in_bounds_load %view0[%c0] : memref<256xf32, #wg>, index -> f32
  // CHECK: rock.live_out %[[ALLOC2]]
  %load1_1 = rock.in_bounds_load %view1[%c0] : memref<256xf32, #wg>, index -> f32
  
  // CHECK: rock.live_in %[[ALLOC3]]
  rock.in_bounds_store %cst_0 -> %view2[%c0] : f32 -> memref<512xf32, #wg>, index
  // CHECK: rock.live_out %[[ALLOC3]]
  %load2_1 = rock.in_bounds_load %view2[%c0] : memref<512xf32, #wg>, index -> f32

  // CHECK-NOT: rock.live_in
  // CHECK-NOT: rock.live_out

  return
}

// Check that we handle ops with memory effects but no associated value instead of crashing
// CHECK-LABEL: func.func @op_with_memory_effects_but_no_associated_value
func.func @op_with_memory_effects_but_no_associated_value() attributes{arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  rock.alloc() : memref<1024xi8, #wg>
  // CHECK: gpu.printf "Let's try to trigger a crash!"
  gpu.printf "Let's try to trigger a crash!"
  // CHECK-NOT: rock.live_in
  // CHECK-NOT: rock.live_out
  return
}