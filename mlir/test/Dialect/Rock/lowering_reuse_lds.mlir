// RUN: rocmlir-opt -rock-reuse-lds %s | FileCheck %s

#wg = #gpu.address_space<workgroup>
#priv = #gpu.address_space<private>


// CHECK-LABEL: func.func @rock_reuse_two
func.func @rock_reuse_two() attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  // CHECK: %[[ALLOC:.*]] = rock.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC]][%[[OFFSET]]][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %0 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK-NOT: rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 1024 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC]][%[[OFFSET2]]][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %1 = rock.alloc() : memref<1024xi8, #wg>
  rock.dealloc %0 : memref<1024xi8, #wg>
  rock.dealloc %1 : memref<1024xi8, #wg>
  
  // CHECK-NOT: rock.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET3:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC]][%[[OFFSET3]]][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<2048xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  %2 = rock.alloc() : memref<2048xi8, #wg>
  // CHECK-NEXT: rock.dealloc %[[ALLOC]]
  rock.dealloc %2 : memref<2048xi8, #wg>

  return
}

// CHECK-LABEL: func.func @rock_reuse_samesize
func.func @rock_reuse_samesize() attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  // CHECK: %[[ALLOC:.*]] = rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC]][%[[OFFSET]]][] : memref<16384xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %0 = rock.alloc() : memref<16384xi8, #wg>
  rock.dealloc %0 : memref<16384xi8, #wg>

  // CHECK-NOT: rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC]][%[[OFFSET2]]][] : memref<16384xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  %1 = rock.alloc() : memref<16384xi8, #wg>
  // CHECK-NEXT: rock.dealloc %[[ALLOC]]
  rock.dealloc %1 : memref<16384xi8, #wg>
  
  return
}

// CHECK-LABEL: func.func @rock_noreuse
func.func @rock_noreuse() attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  // CHECK: %[[ALLOC1:.*]] = rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC2:.*]] = rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC1]][%[[OFFSET]]][] : memref<16384xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %0 = rock.alloc() : memref<16384xi8, #wg>
  // CHECK-NOT: rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC2]][%[[OFFSET2]]][] : memref<16384xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %1 = rock.alloc() : memref<16384xi8, #wg>
  rock.dealloc %0 : memref<16384xi8, #wg>
  // CHECK-NEXT: rock.dealloc %[[ALLOC1]]
  // CHECK-NEXT: rock.dealloc %[[ALLOC2]]
  rock.dealloc %1 : memref<16384xi8, #wg>
  
  return
}

// CHECK-LABEL: func.func @rock_reuse_all
func.func @rock_reuse_all() attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  // CHECK: %[[ALLOC1:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC2:.*]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC1]][%[[OFFSET]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %1 = rock.alloc() : memref<1024xi8, #wg>
  
  // CHECK: %[[OFFSET2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC2]][%[[OFFSET2]]][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %2 = rock.alloc() : memref<1024xi8, #wg>
  
  // CHECK: %[[OFFSET3:.*]] = arith.constant 1024 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC2]][%[[OFFSET3]]][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %3 = rock.alloc() : memref<1024xi8, #wg>
  rock.dealloc %2 : memref<1024xi8, #wg>

  // CHECK: %[[OFFSET4:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC2]][%[[OFFSET4]]][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  %4 = rock.alloc() : memref<1024xi8, #wg>
  rock.dealloc %1 : memref<1024xi8, #wg>
  rock.dealloc %3 : memref<1024xi8, #wg>
  rock.dealloc %4 : memref<1024xi8, #wg>

  // CHECK: %[[OFFSET5:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC2]][%[[OFFSET5]]][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<4096xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  %5 = rock.alloc() : memref<4096xi8, #wg>
  // CHECK: %[[OFFSET6:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC1]][%[[OFFSET6]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %6 = rock.alloc() : memref<1024xi8, #wg>
  rock.dealloc %5 : memref<4096xi8, #wg>
  // CHECK-NEXT: rock.dealloc %[[ALLOC1]]
  // CHECK-NEXT: rock.dealloc %[[ALLOC2]]
  rock.dealloc %6 : memref<1024xi8, #wg>

  return
}

// CHECK-LABEL: func.func @rock_reuse_fragmentation
func.func @rock_reuse_fragmentation() attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  
  // CHECK: %[[ALLOC1:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC2:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC3:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC4:.*]] = rock.alloc() : memref<3072xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC1]][%[[OFFSET]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %1 = rock.alloc() : memref<1024xi8, #wg>
  
  // CHECK: %[[OFFSET2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC2]][%[[OFFSET2]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %2 = rock.alloc() : memref<1024xi8, #wg>
  
  // CHECK: %[[OFFSET3:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC3]][%[[OFFSET3]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %3 = rock.alloc() : memref<1024xi8, #wg>
  rock.dealloc %2 : memref<1024xi8, #wg>

  // CHECK: %[[OFFSET4:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC2]][%[[OFFSET4]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  %4 = rock.alloc() : memref<1024xi8, #wg>
  rock.dealloc %1 : memref<1024xi8, #wg>
  rock.dealloc %4 : memref<1024xi8, #wg>

  // CHECK: %[[OFFSET5:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC4]][%[[OFFSET5]]][] : memref<3072xi8, #gpu.address_space<workgroup>> to memref<3072xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %5 = rock.alloc() : memref<3072xi8, #wg>
  rock.dealloc %3 : memref<1024xi8, #wg>
  // CHECK-NEXT: rock.dealloc %[[ALLOC1]]
  // CHECK-NEXT: rock.dealloc %[[ALLOC2]]
  // CHECK-NEXT: rock.dealloc %[[ALLOC3]]
  // CHECK-NEXT: rock.dealloc %[[ALLOC4]]
  rock.dealloc %5 : memref<3072xi8, #wg>

  return
}

// CHECK-LABEL: func.func @rock_reuse_align
func.func @rock_reuse_align() attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  
  // CHECK: %[[ALLOC1:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC2:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[ALLOC3:.*]] = rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC1]][%[[OFFSET]]][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<1xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %1 = rock.alloc() : memref<1xi8, #wg>
  
  // CHECK: %[[OFFSET2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC2]][%[[OFFSET2]]][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<15xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %2 = rock.alloc() : memref<15xi8, #wg>
  
  // CHECK: %[[OFFSET3:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC3]][%[[OFFSET3]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<1023xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %3 = rock.alloc() : memref<1023xi8, #wg>
  rock.dealloc %2 : memref<15xi8, #wg>

  // CHECK: %[[OFFSET4:.*]] = arith.constant 0 : index
  // CHECK-NEXT: rock.noalias_view %[[ALLOC2]][%[[OFFSET4]]][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<3xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  %4 = rock.alloc() : memref<3xi8, #wg>
  rock.dealloc %1 : memref<1xi8, #wg>
  rock.dealloc %4 : memref<3xi8, #wg>
  // CHECK-NEXT: rock.dealloc %[[ALLOC1]]
  // CHECK-NEXT: rock.dealloc %[[ALLOC2]]
  // CHECK-NEXT: rock.dealloc %[[ALLOC3]]
  rock.dealloc %3 : memref<1023xi8, #wg>

  return
}
