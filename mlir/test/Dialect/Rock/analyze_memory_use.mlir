// RUN: rocmlir-opt -rock-analyze-memory-use %s | FileCheck %s

// Note: the 64-bit index support is tested in large_tensor_detection

// CHECK-LABEL: @base_case
// CHECK-SAME: (%{{.*}}: memref<16xf32> {llvm.align = 16 : i64, llvm.dereferenceable = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly}, %{{.*}}: memref<16xf32> {llvm.align = 16 : i64, llvm.dereferenceable = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.writeonly}, %{{.*}}: index)
func.func @base_case(%arg0: memref<16xf32>, %arg1: memref<16xf32>, %arg2: index) attributes {kernel} {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  %v = rock.global_load %arg0[%arg2] if %true : memref<16xf32> -> vector<4xf32>
  %buf = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
  rock.in_bounds_store %v -> %buf[%c0] : vector<4xf32> -> memref<4xf32, #gpu.address_space<private>>, index
  // CHECK: rock.global_store
  rock.global_store set %buf[%c0] -> %arg1[%arg2] if %true features = none {length = 2 : index} : memref<4xf32, #gpu.address_space<private>> -> memref<16xf32>
  func.return
}

// CHECK-LABEL: @atomic_case
// CHECK-NOT: llvm.writeonly
func.func @atomic_case(%arg0: memref<16xf32>, %arg1: memref<16xf32>, %arg2: index) attributes {kernel} {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  %v = rock.global_load %arg0[%arg2] if %true : memref<16xf32> -> vector<4xf32>
  %buf = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
  rock.in_bounds_store %v -> %buf[%c0] : vector<4xf32> -> memref<4xf32, #gpu.address_space<private>>, index
  rock.global_store atomic_add %buf[%c0] -> %arg1[%arg2] if %true features = none {length = 2 : index} : memref<4xf32, #gpu.address_space<private>> -> memref<16xf32>
  func.return
}

// CHECK-LABEL: @collapse_case
// CHECK-SAME: (%{{.*}}: memref<4x4xf32> {llvm.align = 16 : i64, llvm.dereferenceable = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly}, %{{.*}}: memref<16xf32> {llvm.align = 16 : i64, llvm.dereferenceable = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.writeonly}, %{{.*}}: index)
func.func @collapse_case(%arg0: memref<4x4xf32>, %arg1: memref<16xf32>, %arg2: index) attributes {kernel} {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  %arg0_1 = memref.collapse_shape %arg0 [[0, 1]] : memref<4x4xf32> into memref<16xf32>
  %v = rock.global_load %arg0_1[%arg2] if %true : memref<16xf32> -> vector<4xf32>
  %buf = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
  rock.in_bounds_store %v -> %buf[%c0] : vector<4xf32> -> memref<4xf32, #gpu.address_space<private>>, index
  rock.global_store set %buf[%c0] -> %arg1[%arg2] if %true features = none {length = 2 : index} : memref<4xf32, #gpu.address_space<private>> -> memref<16xf32>
  func.return
}

// CHECK-LABEL: @block_readonly_writeonly
// CHECK-NOT: llvm.readonly
// CHECK-NOT: llvm.writeonly
func.func @block_readonly_writeonly(%arg0: memref<16xf32>, %arg1: memref<16xf32>, %arg2: index) attributes {kernel} {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  // Blocks `readonly`
  %v0 = memref.load %arg0[%arg2] : memref<16xf32>
  %v = rock.global_load %arg0[%arg2] if %true : memref<16xf32> -> vector<4xf32>
  %buf = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
  rock.in_bounds_store %v -> %buf[%c0] : vector<4xf32> -> memref<4xf32, #gpu.address_space<private>>, index
  // Blocks `writeonly`
  memref.store %v0, %arg1[%arg2] : memref<16xf32>
  rock.global_store set %buf[%c0] -> %arg1[%arg2] if %true features = none {length = 2 : index} : memref<4xf32, #gpu.address_space<private>> -> memref<16xf32>
  func.return
}

// CHECK-LABEL: @dead_arg
// CHECK-SAME: llvm.readnone
func.func @dead_arg(%arg0 : memref<16xf32>) attributes {kernel} {
  return
}

// CHECK-LABEL: @dynamic_shape
// CHECK-NOT: llvm.dereferencable
func.func @dynamic_shape(%arg0: memref<?xf32>) attributes {kernel} {
  return
}
