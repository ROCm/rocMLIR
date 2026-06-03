// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-reuse-lds | FileCheck %s

#wg = #gpu.address_space<workgroup>
#priv = #gpu.address_space<private>

// CHECK-LABEL: func.func @rock_reuse_two
func.func @rock_reuse_two() attributes{rock.arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, rock.kernel} {
  // CHECK: %[[ALLOC:.*]] = rock.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET]]][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %0 = rock.alloc() : memref<1024xi8, #wg>
  %1 = rock.alloc() : memref<1024xi8, #wg>
  %2 = rock.alloc() : memref<2048xi8, #wg>
  rock.live_in %0 : memref<1024xi8, #wg>
  // CHECK-NOT: rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 1024 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET2]]][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  rock.live_in %1 : memref<1024xi8, #wg>
  rock.live_out %0 : memref<1024xi8, #wg>
  rock.live_out %1 : memref<1024xi8, #wg>
  
  // CHECK-NOT: rock.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET3:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET3]]][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<2048xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  rock.live_in %2 : memref<2048xi8, #wg>
  rock.live_out %2 : memref<2048xi8, #wg>

  return
}

// CHECK-LABEL: func.func @rock_reuse_samesize
func.func @rock_reuse_samesize() attributes{rock.arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, rock.kernel} {
  // CHECK: %[[ALLOC:.*]] = rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET]]][] : memref<16384xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %0 = rock.alloc() : memref<16384xi8, #wg>
  %1 = rock.alloc() : memref<16384xi8, #wg>
  rock.live_in %0 : memref<16384xi8, #wg>
  rock.live_out %0 : memref<16384xi8, #wg>

  // CHECK-NOT: rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET2]]][] : memref<16384xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  rock.live_in %1 : memref<16384xi8, #wg>
  rock.live_out %1 : memref<16384xi8, #wg>
  
  return
}

// CHECK-LABEL: func.func @rock_noreuse
func.func @rock_noreuse() attributes{rock.arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, rock.kernel} {
  // CHECK: %[[ALLOC1:.*]] = rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC2:.*]] = rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC1]][%[[OFFSET]]][] : memref<16384xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %0 = rock.alloc() : memref<16384xi8, #wg>
  %1 = rock.alloc() : memref<16384xi8, #wg>
  rock.live_in %0 : memref<16384xi8, #wg>
  // CHECK-NOT: rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC2]][%[[OFFSET2]]][] : memref<16384xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  rock.live_in %1 : memref<16384xi8, #wg>
  rock.live_out %0 : memref<16384xi8, #wg>
  rock.live_out %1 : memref<16384xi8, #wg>
  
  return
}

// CHECK-LABEL: func.func @rock_reuse_all
func.func @rock_reuse_all() attributes{rock.arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, rock.kernel} {
  // CHECK: %[[ALLOC1:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC2:.*]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC1]][%[[OFFSET]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %1 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC2]][%[[OFFSET2]]][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %2 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[OFFSET3:.*]] = arith.constant 1024 : index
  // CHECK-NEXT: memref.view %[[ALLOC2]][%[[OFFSET3]]][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %3 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[OFFSET4:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC2]][%[[OFFSET4]]][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %4 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[OFFSET5:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC2]][%[[OFFSET5]]][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<4096xi8, #gpu.address_space<workgroup>>
  %5 = rock.alloc() : memref<4096xi8, #wg>
  // CHECK: %[[OFFSET6:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC1]][%[[OFFSET6]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %6 = rock.alloc() : memref<1024xi8, #wg>
  rock.live_in %1 : memref<1024xi8, #wg>
  rock.live_in %2 : memref<1024xi8, #wg>
  rock.live_in %3 : memref<1024xi8, #wg>
  rock.live_out %2 : memref<1024xi8, #wg>

  // CHECK-NEXT: rock.lds_barrier
  rock.live_in %4 : memref<1024xi8, #wg>
  rock.live_out %1 : memref<1024xi8, #wg>
  rock.live_out %3 : memref<1024xi8, #wg>
  rock.live_out %4 : memref<1024xi8, #wg>

  // CHECK-NEXT: rock.lds_barrier
  rock.live_in %5 : memref<4096xi8, #wg>
  // CHECK-NOT: rock.lds_barrier
  rock.live_in %6 : memref<1024xi8, #wg>
  rock.live_out %5 : memref<4096xi8, #wg>
  rock.live_out %6 : memref<1024xi8, #wg>

  return
}

// CHECK-LABEL: func.func @rock_reuse_fragmentation
func.func @rock_reuse_fragmentation() attributes{rock.arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, rock.kernel} {
  
  // CHECK: %[[ALLOC1:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC2:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC3:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC4:.*]] = rock.alloc() : memref<3072xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC1]][%[[OFFSET]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %1 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC2]][%[[OFFSET2]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %2 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[OFFSET3:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC3]][%[[OFFSET3]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %3 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[OFFSET4:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC2]][%[[OFFSET4]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %4 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[OFFSET5:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC4]][%[[OFFSET5]]][] : memref<3072xi8, #gpu.address_space<workgroup>> to memref<3072xi8, #gpu.address_space<workgroup>>
  %5 = rock.alloc() : memref<3072xi8, #wg>
  rock.live_in %1 : memref<1024xi8, #wg>
  rock.live_in %2 : memref<1024xi8, #wg>
  rock.live_in %3 : memref<1024xi8, #wg>
  rock.live_out %2 : memref<1024xi8, #wg>

  // CHECK-NEXT: rock.lds_barrier
  rock.live_in %4 : memref<1024xi8, #wg>
  rock.live_out %1 : memref<1024xi8, #wg>
  rock.live_out %4 : memref<1024xi8, #wg>

  // CHECK-NOT: rock.lds_barrier
  rock.live_in %5 : memref<3072xi8, #wg>
  rock.live_out %3 : memref<1024xi8, #wg>
  rock.live_out %5 : memref<3072xi8, #wg>

  return
}

// CHECK-LABEL: func.func @rock_reuse_align
func.func @rock_reuse_align() attributes{rock.arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, rock.kernel} {
  // CHECK: %[[ALLOC1:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC2:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC3:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC1]][%[[OFFSET]]][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<1xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %1 = rock.alloc() : memref<1xi8, #wg>
  %2 = rock.alloc() : memref<15xi8, #wg>
  %3 = rock.alloc() : memref<1023xi8, #wg>
  %4 = rock.alloc() : memref<3xi8, #wg>
  rock.live_in %1 : memref<1xi8, #wg>
  
  // CHECK: %[[OFFSET2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC2]][%[[OFFSET2]]][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<15xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  rock.live_in %2 : memref<15xi8, #wg>
  
  // CHECK: %[[OFFSET3:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC3]][%[[OFFSET3]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1023xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  rock.live_in %3 : memref<1023xi8, #wg>
  rock.live_out %2 : memref<15xi8, #wg>

  // CHECK: %[[OFFSET4:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC2]][%[[OFFSET4]]][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<3xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  rock.live_in %4 : memref<3xi8, #wg>
  rock.live_out %1 : memref<1xi8, #wg>
  rock.live_out %4 : memref<3xi8, #wg>
  rock.live_out %3 : memref<1023xi8, #wg>

  return
}

// CHECK-LABEL: func.func @rock_multiple_liveness_ranges
func.func @rock_multiple_liveness_ranges() attributes{rock.arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, rock.kernel} {
  // CHECK: %[[ALLOC:.*]] = rock.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET]]][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %0 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK-NOT: rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 1024 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET2]]][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %1 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK-NOT: rock.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET3:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET3]]][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<2048xi8, #gpu.address_space<workgroup>>
  %2 = rock.alloc() : memref<2048xi8, #wg>
  rock.live_in %0 : memref<1024xi8, #wg>
  rock.live_in %1 : memref<1024xi8, #wg>
  rock.live_out %0 : memref<1024xi8, #wg>
  rock.live_out %1 : memref<1024xi8, #wg>
  
  // CHECK: rock.lds_barrier
  rock.live_in %2 : memref<2048xi8, #wg>
  rock.live_out %2 : memref<2048xi8, #wg>

  // CHECK: rock.lds_barrier
  rock.live_in %0 : memref<1024xi8, #wg>
  rock.live_in %1 : memref<1024xi8, #wg>
  rock.live_out %0 : memref<1024xi8, #wg>
  rock.live_out %1 : memref<1024xi8, #wg>
  
  // CHECK: rock.lds_barrier
  rock.live_in %2 : memref<2048xi8, #wg>
  rock.live_out %2 : memref<2048xi8, #wg>

  // No more barriers
  // CHECK-NOT: rock.lds_barrier

  return
}

// CHECK-LABEL: func.func @rock_multiple_liveness_ranges_interference_sometimes
func.func @rock_multiple_liveness_ranges_interference_sometimes() attributes{rock.arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, rock.kernel} {
  // CHECK: %[[ALLOC1:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  %0 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK: %[[ALLOC2:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  %1 = rock.alloc() : memref<1024xi8, #wg>
  
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC1]][%[[OFFSET]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC2]][%[[OFFSET2]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>

  // no interference
  rock.live_in %0 : memref<1024xi8, #wg>
  rock.live_out %0 : memref<1024xi8, #wg>
  rock.live_in %1 : memref<1024xi8, #wg>
  rock.live_out %1 : memref<1024xi8, #wg>

  // interference
  rock.live_in %0 : memref<1024xi8, #wg>
  rock.live_in %1 : memref<1024xi8, #wg>
  rock.live_out %0 : memref<1024xi8, #wg>
  rock.live_out %1 : memref<1024xi8, #wg>

  // No barriers
  // CHECK-NOT: rock.lds_barrier

  return
}

// CHECK-LABEL: func.func @rock_multiple_liveness_ranges_interference_sometimes2
func.func @rock_multiple_liveness_ranges_interference_sometimes2() attributes{rock.arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, rock.kernel} {
  %0 = rock.alloc() : memref<1024xi8, #wg>
  %1 = rock.alloc() : memref<1024xi8, #wg>
  %2 = rock.alloc() : memref<512xi8, #wg>
  // CHECK: %[[ALLOC1:.*]] = rock.alloc() : memref<512xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC2:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>

  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC2]][%[[OFFSET]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC2]][%[[OFFSET2]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET3:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC1]][%[[OFFSET3]]][] : memref<512xi8, #gpu.address_space<workgroup>> to memref<512xi8, #gpu.address_space<workgroup>>

  // no interference
  rock.live_in %0 : memref<1024xi8, #wg>
  rock.live_out %0 : memref<1024xi8, #wg>
  // CHECK: rock.lds_barrier
  rock.live_in %1 : memref<1024xi8, #wg>
  rock.live_out %1 : memref<1024xi8, #wg>
  rock.live_in %2 : memref<512xi8, #wg>
  rock.live_out %2 : memref<512xi8, #wg>

  // interference
  rock.live_in %2 : memref<512xi8, #wg>
  // CHECK: rock.lds_barrier
  rock.live_in %0 : memref<1024xi8, #wg>
  rock.live_out %0 : memref<1024xi8, #wg>
  // CHECK: rock.lds_barrier
  rock.live_in %1 : memref<1024xi8, #wg>
  rock.live_out %1 : memref<1024xi8, #wg>
  rock.live_out %2 : memref<512xi8, #wg>

  // No barriers
  // CHECK-NOT: rock.lds_barrier

  return
}

// CHECK-LABEL: func.func @rock_multiple_liveness_ranges_interference_sometimes3
func.func @rock_multiple_liveness_ranges_interference_sometimes3() attributes{rock.arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, rock.kernel} {
  %0 = rock.alloc() : memref<1024xi8, #wg>
  %1 = rock.alloc() : memref<1024xi8, #wg>
  %2 = rock.alloc() : memref<512xi8, #wg>
  // CHECK: %[[ALLOC1:.*]] = rock.alloc() : memref<512xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC2:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC3:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>

  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC2]][%[[OFFSET]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC3]][%[[OFFSET2]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET3:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC1]][%[[OFFSET3]]][] : memref<512xi8, #gpu.address_space<workgroup>> to memref<512xi8, #gpu.address_space<workgroup>>

  // no interference
  rock.live_in %0 : memref<1024xi8, #wg>
  rock.live_out %0 : memref<1024xi8, #wg>
  rock.live_in %1 : memref<1024xi8, #wg>
  rock.live_out %1 : memref<1024xi8, #wg>
  rock.live_in %2 : memref<512xi8, #wg>
  rock.live_out %2 : memref<512xi8, #wg>

  // interference
  rock.live_in %2 : memref<512xi8, #wg>
  rock.live_in %1 : memref<1024xi8, #wg>
  rock.live_in %0 : memref<1024xi8, #wg>
  rock.live_out %0 : memref<1024xi8, #wg>
  rock.live_out %1 : memref<1024xi8, #wg>
  rock.live_out %2 : memref<512xi8, #wg>

  // No barriers
  // CHECK-NOT: rock.lds_barrier

  return
}
