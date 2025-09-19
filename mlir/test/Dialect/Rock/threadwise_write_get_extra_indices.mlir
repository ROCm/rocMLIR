// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: getExtraIndices returns the extraIndices for the dest operand
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @basic_extra_indices
func.func @basic_extra_indices(%src: memref<16xf32, 5>, %dst: memref<16xf32, 0>, %idx0: index) {
  // Only the second operand (%dst) is an accepting view operand.
  // CHECK: "test.get_extra_indices"(%src, %dst, %idx0)
  "test.get_extra_indices"(%src, %dst, %idx0) : (memref<16xf32, 5>, memref<16xf32, 0>, index) -> index
  return
}

//===----------------------------------------------------------------------===//
// Test: getExtraIndices returns the extraIndices for the dest operand with multiple indices
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @multi_extra_indices
func.func @multi_extra_indices(%src: memref<8x8xf16, 5>, %dst: memref<8x8xf16, 0>, %idx0: index, %idx1: index) {
  // Only the second operand (%dst) is an accepting view operand.
  // CHECK: "test.get_extra_indices"(%src, %dst, %idx0, %idx1)
  "test.get_extra_indices"(%src, %dst, %idx0, %idx1) : (memref<8x8xf16, 5>, memref<8x8xf16, 0>, index, index) -> (index, index)
  return
}

//===----------------------------------------------------------------------===//
// Test: getExtraIndices returns nothing for the source operand (not accepting view)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @src_operand_no_extra_indices
func.func @src_operand_no_extra_indices(%src: memref<4x4xi8, 5>, %dst: memref<4x4xi32, 0>, %idx: index) {
  // Only the second operand (%dst) is an accepting view operand.
  // CHECK: "test.get_extra_indices"(%src)
  "test.get_extra_indices"(%src) : (memref<4x4xi8, 5>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test: getExtraIndices returns nothing for an unrelated operand
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @unrelated_operand
func.func @unrelated_operand(%src: memref<2xvector<4xf32>, 5>, %dst: memref<2xvector<4xf32>, 0>, %other: i32) {
  // Only the second operand (%dst) is an accepting view operand.
  // CHECK: "test.get_extra_indices"(%other)
  "test.get_extra_indices"(%other) : (i32) -> ()
  return
}