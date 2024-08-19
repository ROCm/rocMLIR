// RUN: rocmlir-opt -rock-reuse-lds %s | FileCheck %s

#wg = #gpu.address_space<workgroup>
#priv = #gpu.address_space<private>


// CHECK-LABEL: func.func @rock_reuse_two
func.func @rock_reuse_two() attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  // CHECK: %[[ALLOC:.*]] = rock.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET]]][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %0 = rock.alloc() : memref<1024xi8, #wg>
  // CHECK-NOT: rock.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 1024 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET2]]][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %1 = rock.alloc() : memref<1024xi8, #wg>
  rock.dealloc %0 : memref<1024xi8, #wg>
  rock.dealloc %1 : memref<1024xi8, #wg>
  
  // CHECK-NOT: rock.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET3:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET3]]][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<2048xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  %2 = rock.alloc() : memref<2048xi8, #wg>
  rock.dealloc %2 : memref<2048xi8, #wg>

  return
}

// CHECK-LABEL: func.func @rock_reuse_samesize
func.func @rock_reuse_samesize() attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  // CHECK: %[[ALLOC:.*]] = rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET]]][] : memref<16384xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
  %0 = rock.alloc() : memref<16384xi8, #wg>
  rock.dealloc %0 : memref<16384xi8, #wg>

  // CHECK-NOT: rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET2]]][] : memref<16384xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  %1 = rock.alloc() : memref<16384xi8, #wg>
  rock.dealloc %1 : memref<16384xi8, #wg>
  
  return
}

// CHECK-LABEL: func.func @rock_noreuse
func.func @rock_noreuse() attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  // CHECK: %[[ALLOC:.*]] = rock.alloc() : memref<32768xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET]]][] : memref<32768xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
  %0 = rock.alloc() : memref<16384xi8, #wg>
  // CHECK-NOT: rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 16384 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET2]]][] : memref<32768xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %1 = rock.alloc() : memref<16384xi8, #wg>
  rock.dealloc %0 : memref<16384xi8, #wg>
  rock.dealloc %1 : memref<16384xi8, #wg>
  
  return
}

// CHECK-LABEL: func.func @rock_reuse_all
func.func @rock_reuse_all() attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  
  // CHECK: %[[ALLOC:.*]] = rock.alloc() : memref<5120xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET]]][] : memref<5120xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %1 = rock.alloc() : memref<1024xi8, #wg>
  
  // CHECK: %[[OFFSET2:.*]] = arith.constant 1024 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET2]]][] : memref<5120xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %2 = rock.alloc() : memref<1024xi8, #wg>
  
  // CHECK: %[[OFFSET3:.*]] = arith.constant 2048 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET3]]][] : memref<5120xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %3 = rock.alloc() : memref<1024xi8, #wg>
  rock.dealloc %2 : memref<1024xi8, #wg>

  // CHECK: %[[OFFSET4:.*]] = arith.constant 1024 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET4]]][] : memref<5120xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  %4 = rock.alloc() : memref<1024xi8, #wg>
  rock.dealloc %1 : memref<1024xi8, #wg>
  rock.dealloc %3 : memref<1024xi8, #wg>
  rock.dealloc %4 : memref<1024xi8, #wg>

  // CHECK: %[[OFFSET5:.*]] = arith.constant 1024 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET5]]][] : memref<5120xi8, #gpu.address_space<workgroup>> to memref<4096xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  %5 = rock.alloc() : memref<4096xi8, #wg>
  // CHECK: %[[OFFSET6:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET6]]][] : memref<5120xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %6 = rock.alloc() : memref<1024xi8, #wg>
  rock.dealloc %5 : memref<4096xi8, #wg>
  rock.dealloc %6 : memref<1024xi8, #wg>

  return
}

// CHECK-LABEL: func.func @rock_reuse_fragmentation
func.func @rock_reuse_fragmentation() attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  
  // CHECK: %[[ALLOC:.*]] = rock.alloc() : memref<6144xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET]]][] : memref<6144xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %1 = rock.alloc() : memref<1024xi8, #wg>
  
  // CHECK: %[[OFFSET2:.*]] = arith.constant 1024 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET2]]][] : memref<6144xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %2 = rock.alloc() : memref<1024xi8, #wg>
  
  // CHECK: %[[OFFSET3:.*]] = arith.constant 2048 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET3]]][] : memref<6144xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  %3 = rock.alloc() : memref<1024xi8, #wg>
  rock.dealloc %2 : memref<1024xi8, #wg>

  // CHECK: %[[OFFSET4:.*]] = arith.constant 1024 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET4]]][] : memref<6144xi8, #gpu.address_space<workgroup>> to memref<1024xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  %4 = rock.alloc() : memref<1024xi8, #wg>
  rock.dealloc %1 : memref<1024xi8, #wg>
  rock.dealloc %4 : memref<1024xi8, #wg>

  // CHECK: %[[OFFSET5:.*]] = arith.constant 3072 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET5]]][] : memref<6144xi8, #gpu.address_space<workgroup>> to memref<3072xi8, #gpu.address_space<workgroup>>
  // CHECK-NOT: rock.lds_barrier
  %5 = rock.alloc() : memref<3072xi8, #wg>
  rock.dealloc %3 : memref<1024xi8, #wg>
  rock.dealloc %5 : memref<3072xi8, #wg>

  return
}

// CHECK-LABEL: func.func @rock_reuse_align
func.func @rock_reuse_align() attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  
  // CHECK: %[[ALLOC:.*]] = rock.alloc() : memref<1056xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET]]][] : memref<1056xi8, #gpu.address_space<workgroup>> to memref<1xi8, #gpu.address_space<workgroup>>
  %1 = rock.alloc() : memref<1xi8, #wg>
  
  // CHECK: %[[OFFSET2:.*]] = arith.constant 16 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET2]]][] : memref<1056xi8, #gpu.address_space<workgroup>> to memref<15xi8, #gpu.address_space<workgroup>>
  %2 = rock.alloc() : memref<15xi8, #wg>
  
  // CHECK: %[[OFFSET3:.*]] = arith.constant 32 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET3]]][] : memref<1056xi8, #gpu.address_space<workgroup>> to memref<1023xi8, #gpu.address_space<workgroup>>
  %3 = rock.alloc() : memref<1023xi8, #wg>
  rock.dealloc %2 : memref<15xi8, #wg>

  // CHECK: %[[OFFSET4:.*]] = arith.constant 16 : index
  // CHECK-NEXT: memref.view %[[ALLOC]][%[[OFFSET4]]][] : memref<1056xi8, #gpu.address_space<workgroup>> to memref<3xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: rock.lds_barrier
  %4 = rock.alloc() : memref<3xi8, #wg>
  rock.dealloc %1 : memref<1xi8, #wg>
  rock.dealloc %4 : memref<3xi8, #wg>
  rock.dealloc %3 : memref<1023xi8, #wg>

  return
}
