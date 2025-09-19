// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: 2D, no transpose, standard case
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @basic_2d
func.func @basic_2d(%a: memref<2x4xf32>, %b: memref<2x4xf32>, %c: memref<2x4xf32>, %out: memref<2x4xf32>) {
  %op = "rock.gemm_elementwise_gemm"(%a, %b, %c, %out) : (memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>) -> memref<2x4xf32>
  // CHECK: "test.get_gemm_gemm_size"(%op)
  "test.get_gemm_gemm_size"(%op) : (operation) -> !rock.gemm_gemm_size
  return
}

//===----------------------------------------------------------------------===//
// Test: 3D, batch dimension present
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @batch_3d
func.func @batch_3d(%a: memref<3x4x5xf16>, %b: memref<3x5x6xf16>, %c: memref<3x6x7xf16>, %out: memref<3x4x7xf16>) {
  %op = "rock.gemm_elementwise_gemm"(%a, %b, %c, %out) : (memref<3x4x5xf16>, memref<3x5x6xf16>, memref<3x6x7xf16>, memref<3x4x7xf16>) -> memref<3x4x7xf16>
  // CHECK: "test.get_gemm_gemm_size"(%op)
  "test.get_gemm_gemm_size"(%op) : (operation) -> !rock.gemm_gemm_size
  return
}

//===----------------------------------------------------------------------===//
// Test: 2D, transposed C and output
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @transposed_c_out
func.func @transposed_c_out(%a: memref<2x4xf32>, %b: memref<2x4xf32>, %c: memref<2x4xf32>, %out: memref<2x4xf32>) {
  %op = "rock.gemm_elementwise_gemm"(%a, %b, %c, %out) {c_transposed, o_transposed} : (memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>) -> memref<2x4xf32>
  // CHECK: "test.get_gemm_gemm_size"(%op)
  "test.get_gemm_gemm_size"(%op) : (operation) -> !rock.gemm_gemm_size
  return
}

//===----------------------------------------------------------------------===//
// Test: 3D, transposed A, B, C, and output
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @all_transposed
func.func @all_transposed(%a: memref<3x5x4xf32>, %b: memref<3x6x5xf32>, %c: memref<3x7x6xf32>, %out: memref<3x7x4xf32>) {
  %op = "rock.gemm_elementwise_gemm"(%a, %b, %c, %out) {a_transposed, b_transposed, c_transposed, o_transposed} : (memref<3x5x4xf32>, memref<3x6x5xf32>, memref<3x7x6xf32>, memref<3x7x4xf32>) -> memref<3x7x4xf32>
  // CHECK: "test.get_gemm_gemm_size"(%op)
  "test.get_gemm_gemm_size"(%op) : (operation) -> !rock.gemm_gemm_size
  return
}

//===----------------------------------------------------------------------===//
// Test: 2D, mismatched shapes (should still call getGemmGemmSize, but may error elsewhere)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @mismatched_shapes
func.func @mismatched_shapes(%a: memref<2x4xf32>, %b: memref<2x5xf32>, %c: memref<2x6xf32>, %out: memref<2x7xf32>) {
  %op = "rock.gemm_elementwise_gemm"(%a, %b, %c, %out) : (memref<2x4xf32>, memref<2x5xf32>, memref<2x6xf32>, memref<2x7xf32>) -> memref<2x7xf32>
  // CHECK: "test.get_gemm_gemm_size"(%op)
  "test.get_gemm_gemm_size"(%op) : (operation) -> !rock.gemm_gemm_size
  return
}