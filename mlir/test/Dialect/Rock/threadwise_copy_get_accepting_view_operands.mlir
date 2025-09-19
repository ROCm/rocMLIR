// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: getAcceptingViewOperands returns both source and dest operands
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @accepting_view_operands_basic
func.func @accepting_view_operands_basic(%src: memref<16xf32, 5>, %dst: memref<16xf32, 0>) {
  // Both operands are accepting view operands when no extra indices.
  // CHECK: "test.get_accepting_view_operands"(%src, %dst)
  "test.get_accepting_view_operands"(%src, %dst) : (memref<16xf32, 5>, memref<16xf32, 0>) -> (i1, i1)
  return
}

//===----------------------------------------------------------------------===//
// Test: getAcceptingViewOperands with extraIndicesSource (source has extra indices)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @accepting_view_operands_extra_source
func.func @accepting_view_operands_extra_source(%src: memref<8x8xf16, 5>, %dst: memref<8x8xf16, 0>, %idx: index) {
  // Both operands are accepting view operands; source has extra indices.
  // CHECK: "test.get_accepting_view_operands"(%src, %dst, %idx)
  "test.get_accepting_view_operands"(%src, %dst, %idx) : (memref<8x8xf16, 5>, memref<8x8xf16, 0>, index) -> (i1, i1)
  return
}

//===----------------------------------------------------------------------===//
// Test: getAcceptingViewOperands with extraIndicesDest (dest has extra indices)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @accepting_view_operands_extra_dest
func.func @accepting_view_operands_extra_dest(%src: memref<4x4xi8, 5>, %dst: memref<4x4xi32, 0>, %idx: index) {
  // Both operands are accepting view operands; dest has extra indices.
  // CHECK: "test.get_accepting_view_operands"(%src, %dst, %idx)
  "test.get_accepting_view_operands"(%src, %dst, %idx) : (memref<4x4xi8, 5>, memref<4x4xi32, 0>, index) -> (i1, i1)
  return
}

//===----------------------------------------------------------------------===//
// Test: getAcceptingViewOperands with both extraIndicesSource and extraIndicesDest
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @accepting_view_operands_both_extra
func.func @accepting_view_operands_both_extra(%src: memref<2x2xf32, 5>, %dst: memref<2x2xf32, 0>, %idx0: index, %idx1: index) {
  // Both operands are accepting view operands; both have extra indices.
  // CHECK: "test.get_accepting_view_operands"(%src, %dst, %idx0, %idx1)
  "test.get_accepting_view_operands"(%src, %dst, %idx0, %idx1) : (memref<2x2xf32, 5>, memref<2x2xf32, 0>, index, index) -> (i1, i1)
  return
}

//===----------------------------------------------------------------------===//
// Test: getAcceptingViewOperands with vector element types
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @accepting_view_operands_vector
func.func @accepting_view_operands_vector(%src: memref<2xvector<4xf32>, 5>, %dst: memref<2xvector<4xf32>, 0>) {
  // Both operands are accepting view operands.
  // CHECK: "test.get_accepting_view_operands"(%src, %dst)
  "test.get_accepting_view_operands"(%src, %dst) : (memref<2xvector<4xf32>, 5>, memref<2xvector<4xf32>, 0>) -> (i1, i1)
  return
}