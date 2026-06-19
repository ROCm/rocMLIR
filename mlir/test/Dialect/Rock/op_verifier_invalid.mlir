// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// COM: Negative coverage for assorted small op verifiers in
// COM: mlir/lib/Dialect/Rock/IR/RockDialect.cpp (InsertSliceOp, GpuAllocOp,
// COM: LiveInOp, LiveOutOp, GlobalStoreOp, InBoundsLoadOp, InBoundsStoreOp).

// COM: InsertSliceOp: the source slice cannot be longer than the destination
func.func @insert_slice_too_long(%u: vector<32xf32>, %v: vector<4xf32>) -> vector<4xf32> {
  %i = arith.constant 0 : index
  // expected-error @+1 {{which is longer than destinanation's vector length}}
  %w = rock.insert_slice %u -> %v[%i] : vector<32xf32> -> vector<4xf32>
  return %w : vector<4xf32>
}

// -----

// COM: GpuAllocOp: zero-byte allocations are rejected
func.func @alloc_zero_size() {
  // expected-error @+1 {{The size of rock.alloc should be greather than zero.}}
  %0 = rock.alloc() : memref<0xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// COM: LiveInOp: the operand must come from a rock.alloc
func.func @live_in_not_alloc() {
  %0 = memref.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // expected-error @+1 {{The operand of rock.live_in must be the result of a rock.alloc operation.}}
  rock.live_in %0 : memref<1024xi8, #gpu.address_space<workgroup>>
  return
}

// -----

// COM: LiveInOp: the operand must live in LDS (workgroup) memory
func.func @live_in_not_lds() {
  %0 = rock.alloc() : memref<64xf32, #gpu.address_space<private>>
  // expected-error @+1 {{The operand of rock.live_in must be an LDS memref}}
  rock.live_in %0 : memref<64xf32, #gpu.address_space<private>>
  return
}

// -----

// COM: LiveOutOp: the operand must come from a rock.alloc
func.func @live_out_not_alloc() {
  %0 = memref.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
  // expected-error @+1 {{The operand of rock.live_out must be the result of a rock.alloc operation.}}
  rock.live_out %0 : memref<1024xi8, #gpu.address_space<workgroup>>
  return
}

// -----

// COM: LiveOutOp: the operand must live in LDS (workgroup) memory
func.func @live_out_not_lds() {
  %0 = rock.alloc() : memref<64xf32, #gpu.address_space<private>>
  // expected-error @+1 {{The operand of rock.live_out must be an LDS memref}}
  rock.live_out %0 : memref<64xf32, #gpu.address_space<private>>
  return
}

// -----

// COM: InBoundsLoadOp: coordinate count must match the source rank
func.func @in_bounds_load_wrong_coords(%buffer: memref<128x128xf32, 3>, %idx0: index) -> vector<4xf32> {
  // expected-error @+1 {{Expected 2 coordinates for load}}
  %ret = rock.in_bounds_load %buffer[%idx0] : memref<128x128xf32, 3>, index -> vector<4xf32>
  return %ret : vector<4xf32>
}

// -----

// COM: InBoundsStoreOp: coordinate count must match the destination rank
func.func @in_bounds_store_wrong_coords(%buffer: memref<128x128xf32, 3>, %data: vector<4xf32>, %idx0: index) {
  // expected-error @+1 {{Expected 2 coordinates for store}}
  rock.in_bounds_store %data -> %buffer[%idx0] : vector<4xf32> -> memref<128x128xf32, 3>, index
  return
}

// -----

// COM: GlobalStoreOp: coordinate count must match the destination rank
func.func @global_store_wrong_coords(%source: memref<32xf32, #gpu.address_space<private>>, %dest: memref<?x?x?x?x?xf32>, %valid: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{Expected 5 coordinates for store}}
  rock.global_store set %source[%c0] -> %dest[%c1, %c1, %c1] if %valid {length = 1 : index} : memref<32xf32, #gpu.address_space<private>> -> memref<?x?x?x?x?xf32>
  return
}

// -----

// COM: GlobalStoreOp: the destination must live in global memory
func.func @global_store_not_global(%source: memref<32xf32, #gpu.address_space<private>>, %dest: memref<8xf32, #gpu.address_space<workgroup>>, %valid: i1) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Destination memref must live in global memory}}
  rock.global_store set %source[%c0] -> %dest[%c0] if %valid {length = 1 : index} : memref<32xf32, #gpu.address_space<private>> -> memref<8xf32, #gpu.address_space<workgroup>>
  return
}
