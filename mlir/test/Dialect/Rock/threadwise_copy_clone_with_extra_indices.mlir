// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: cloneWithExtraIndices on source operand (with/without extra indices)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @clone_on_source
func.func @clone_on_source(%src: memref<16xf32, 5>, %dst: memref<16xf32, 0>, %idx0: index, %new_idx: index) {
  // Original op: source has one extra index
  %op = "rock.threadwise_copy"(%src, %idx0, %dst) : (memref<16xf32, 5>, index, memref<16xf32, 0>) -> ()
  // Clone with new extra indices for the source operand
  // CHECK: "test.clone_with_extra_indices"(%op, %src, %new_idx, %dst)
  "test.clone_with_extra_indices"(%op, %src, %new_idx, %dst) : (operation, memref<16xf32, 5>, index, memref<16xf32, 0>) -> operation
  return
}

//===----------------------------------------------------------------------===//
// Test: cloneWithExtraIndices on dest operand (with/without extra indices)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @clone_on_dest
func.func @clone_on_dest(%src: memref<8x8xf16, 5>, %dst: memref<8x8xf16, 0>, %idx0: index, %new_idx: index) {
  // Original op: dest has one extra index
  %op = "rock.threadwise_copy"(%src, %dst, %idx0) : (memref<8x8xf16, 5>, memref<8x8xf16, 0>, index) -> ()
  // Clone with new extra indices for the dest operand
  // CHECK: "test.clone_with_extra_indices"(%op, %src, %dst, %new_idx)
  "test.clone_with_extra_indices"(%op, %src, %dst, %new_idx) : (operation, memref<8x8xf16, 5>, memref<8x8xf16, 0>, index) -> operation
  return
}

//===----------------------------------------------------------------------===//
// Test: cloneWithExtraIndices on source operand with multiple extra indices
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @clone_multi_extra_indices_source
func.func @clone_multi_extra_indices_source(%src: memref<4x4xi8, 5>, %dst: memref<4x4xi32, 0>, %idx0: index, %idx1: index, %new_idx0: index, %new_idx1: index) {
  // Original op: source has two extra indices
  %op = "rock.threadwise_copy"(%src, %idx0, %idx1, %dst) : (memref<4x4xi8, 5>, index, index, memref<4x4xi32, 0>) -> ()
  // Clone with new extra indices for the source operand
  // CHECK: "test.clone_with_extra_indices"(%op, %src, %new_idx0, %new_idx1, %dst)
  "test.clone_with_extra_indices"(%op, %src, %new_idx0, %new_idx1, %dst) : (operation, memref<4x4xi8, 5>, index, index, memref<4x4xi32, 0>) -> operation
  return
}

//===----------------------------------------------------------------------===//
// Test: cloneWithExtraIndices on dest operand with multiple extra indices
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @clone_multi_extra_indices_dest
func.func @clone_multi_extra_indices_dest(%src: memref<2x2xf32, 5>, %dst: memref<2x2xf32, 0>, %idx0: index, %idx1: index, %new_idx0: index, %new_idx1: index) {
  // Original op: dest has two extra indices
  %op = "rock.threadwise_copy"(%src, %dst, %idx0, %idx1) : (memref<2x2xf32, 5>, memref<2x2xf32, 0>, index, index) -> ()
  // Clone with new extra indices for the dest operand
  // CHECK: "test.clone_with_extra_indices"(%op, %src, %dst, %new_idx0, %new_idx1)
  "test.clone_with_extra_indices"(%op, %src, %dst, %new_idx0, %new_idx1) : (operation, memref<2x2xf32, 5>, memref<2x2xf32, 0>, index, index) -> operation
  return
}

//===----------------------------------------------------------------------===//
// Test: cloneWithExtraIndices on unrelated operand returns original op
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @clone_on_unrelated
func.func @clone_on_unrelated(%src: memref<2xvector<4xf32>, 5>, %dst: memref<2xvector<4xf32>, 0>, %other: i32, %new_idx: index) {
  %op = "rock.threadwise_copy"(%src, %dst) : (memref<2xvector<4xf32>, 5>, memref<2xvector<4xf32>, 0>) -> ()
  // Attempt to clone with new extra indices for an unrelated operand (should return original op)
  // CHECK: "test.clone_with_extra_indices"(%op, %other, %src, %new_idx)
  "test.clone_with_extra_indices"(%op, %other, %src, %new_idx) : (operation, i32, memref<2xvector<4xf32>, 5>, index) -> operation
  return
}