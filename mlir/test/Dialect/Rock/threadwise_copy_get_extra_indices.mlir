// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: getExtraIndices returns extraIndicesSource for source operand
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @extra_indices_source
func.func @extra_indices_source(%src: memref<16xf32, 5>, %dst: memref<16xf32, 0>, %idx0: index) {
  // Only the source operand (%src) is checked for extraIndicesSource.
  // CHECK: "test.get_extra_indices"(%src, %dst, %idx0)
  "test.get_extra_indices"(%src, %dst, %idx0) : (memref<16xf32, 5>, memref<16xf32, 0>, index) -> index
  return
}

//===----------------------------------------------------------------------===//
// Test: getExtraIndices returns extraIndicesDest for dest operand
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @extra_indices_dest
func.func @extra_indices_dest(%src: memref<8x8xf16, 5>, %dst: memref<8x8xf16, 0>, %idx0: index) {
  // Only the dest operand (%dst) is checked for extraIndicesDest.
  // CHECK: "test.get_extra_indices"(%src, %dst, %idx0)
  "test.get_extra_indices"(%src, %dst, %idx0) : (memref<8x8xf16, 5>, memref<8x8xf16, 0>, index) -> index
  return
}

//===----------------------------------------------------------------------===//
// Test: getExtraIndices returns multiple extra indices for source operand
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @multi_extra_indices_source
func.func @multi_extra_indices_source(%src: memref<4x4xi8, 5>, %dst: memref<4x4xi32, 0>, %idx0: index, %idx1: index) {
  // Source operand (%src) with multiple extra indices.
  // CHECK: "test.get_extra_indices"(%src, %dst, %idx0, %idx1)
  "test.get_extra_indices"(%src, %dst, %idx0, %idx1) : (memref<4x4xi8, 5>, memref<4x4xi32, 0>, index, index) -> (index, index)
  return
}

//===----------------------------------------------------------------------===//
// Test: getExtraIndices returns multiple extra indices for dest operand
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @multi_extra_indices_dest
func.func @multi_extra_indices_dest(%src: memref<2x2xf32, 5>, %dst: memref<2x2xf32, 0>, %idx0: index, %idx1: index) {
  // Dest operand (%dst) with multiple extra indices.
  // CHECK: "test.get_extra_indices"(%src, %dst, %idx0, %idx1)
  "test.get_extra_indices"(%src, %dst, %idx0, %idx1) : (memref<2x2xf32, 5>, memref<2x2xf32, 0>, index, index) -> (index, index)
  return
}

//===----------------------------------------------------------------------===//
// Test: getExtraIndices returns nothing for unrelated operand
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @unrelated_operand
func.func @unrelated_operand(%src: memref<2xvector<4xf32>, 5>, %dst: memref<2xvector<4xf32>, 0>, %other: i32) {
  // Unrelated operand (%other) should return nothing.
  // CHECK: "test.get_extra_indices"(%other)
  "test.get_extra_indices"(%other) : (i32) -> ()
  return
}