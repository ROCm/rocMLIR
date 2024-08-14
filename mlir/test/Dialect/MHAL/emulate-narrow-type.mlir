// RUN: rocmlir-opt -mhal-emulate-narrow-type -canonicalize \
// RUN:   --split-input-file %s \
// RUN: | FileCheck %s
// COM: This test is up here because MHal doesn't have its own test directory.

// CHECK-LABEL: func.func @foo
// CHECK-SAME memref<4xi8>
// CHECK-COUNT-4: memref.atomic_rmw
func.func @foo(%arg0: memref<8xi4>) -> memref<8xi4> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_i4 = arith.constant 1 : i4
  %c15_i4 = arith.constant 15 : i4
  memref.store %c1_i4, %arg0[%c0] : memref<8xi4>
  memref.store %c15_i4, %arg0[%c1] : memref<8xi4>
  return %arg0 : memref<8xi4>
}

// CHECK-LABEL: func.func @foo_wrapper
// CHECK-SAME: (%[[ARG0:.+]]: memref<4xi8>)
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<4xi8>
// CHECK-NEXT: %[[TOKEN:.+]], %[[V:.+]] = mhal.launch @foo (%[[ALLOC]]) : (memref<4xi8>) -> memref<4xi8>
// CHECK-NEXT: mhal.await %[[TOKEN]]
// CHECK-NEXT: memref.copy %[[V]], %[[ARG0]]
func.func @foo_wrapper(%arg0: memref<8xi4>) {
  %alloc = memref.alloc() : memref<8xi4>
  %token, %v = mhal.launch @foo(%alloc) : (memref<8xi4>) -> memref<8xi4>
  mhal.await %token : !mhal.token
  memref.copy %v, %arg0 : memref<8xi4> to memref<8xi4>
  return
}
