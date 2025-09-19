// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: getAcceptingViewOperands returns only the second operand (dest)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @accepting_view_operand_basic
func.func @accepting_view_operand_basic(%src: memref<16xf32, 5>, %dst: memref<16xf32, 0>) {
  // The second operand (%dst) is the only accepting view operand.
  // CHECK: "test.get_accepting_view_operands"(%src, %dst)
  "test.get_accepting_view_operands"(%src, %dst) : (memref<16xf32, 5>, memref<16xf32, 0>) -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: getAcceptingViewOperands with extraViews and extraIndices
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @accepting_view_operand_extra
func.func @accepting_view_operand_extra(%src: memref<8x8xf16, 5>, %dst: memref<8x8xf16, 0>, %idx: index) {
  // The second operand (%dst) is still the only accepting view operand.
  // CHECK: "test.get_accepting_view_operands"(%src, %dst, %idx)
  "test.get_accepting_view_operands"(%src, %dst, %idx) : (memref<8x8xf16, 5>, memref<8x8xf16, 0>, index) -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: getAcceptingViewOperands with different element types
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @accepting_view_operand_types
func.func @accepting_view_operand_types(%src: memref<4x4xi8, 5>, %dst: memref<4x4xi32, 0>) {
  // The second operand (%dst) is the only accepting view operand.
  // CHECK: "test.get_accepting_view_operands"(%src, %dst)
  "test.get_accepting_view_operands"(%src, %dst) : (memref<4x4xi8, 5>, memref<4x4xi32, 0>) -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: getAcceptingViewOperands with vector element types
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @accepting_view_operand_vector
func.func @accepting_view_operand_vector(%src: memref<2xvector<4xf32>, 5>, %dst: memref<2xvector<4xf32>, 0>) {
  // The second operand (%dst) is the only accepting view operand.
  // CHECK: "test.get_accepting_view_operands"(%src, %dst)
  "test.get_accepting_view_operands"(%src, %dst) : (memref<2xvector<4xf32>, 5>, memref<2xvector<4xf32>, 0>) -> i1
  return
}