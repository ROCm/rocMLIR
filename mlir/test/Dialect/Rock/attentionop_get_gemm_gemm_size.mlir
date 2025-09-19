// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: 2D (no batch), no transposes
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @basic_2d
func.func @basic_2d(%q: memref<4x5xf32>, %k: memref<5x6xf32>, %v: memref<6x7xf32>, %out: memref<4x7xf32>) {
  %op = "rock.attention"(%q, %k, %v, %out) : (memref<4x5xf32>, memref<5x6xf32>, memref<6x7xf32>, memref<4x7xf32>) -> memref<4x7xf32>
  // CHECK: "test.get_gemm_gemm_size"(%op)
  "test.get_gemm_gemm_size"(%op) : (operation) -> !rock.gemm_gemm_size
  return
}

//===----------------------------------------------------------------------===//
// Test: 3D (batched), no transposes
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @batched_3d
func.func @batched_3d(%q: memref<3x4x5xf16>, %k: memref<3x5x6xf16>, %v: memref<3x6x7xf16>, %out: memref<3x4x7xf16>) {
  %op = "rock.attention"(%q, %k, %v, %out) : (memref<3x4x5xf16>, memref<3x5x6xf16>, memref<3x6x7xf16>, memref<3x4x7xf16>) -> memref<3x4x7xf16>
  // CHECK: "test.get_gemm_gemm_size"(%op)
  "test.get_gemm_gemm_size"(%op) : (operation) -> !rock.gemm_gemm_size
  return
}

//===----------------------------------------------------------------------===//
// Test: 2D, all transposed
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @all_transposed_2d
func.func @all_transposed_2d(%q: memref<5x4xf32>, %k: memref<6x5xf32>, %v: memref<7x6xf32>, %out: memref<7x4xf32>) {
  %op = "rock.attention"(%q, %k, %v, %out) {q_transposed, k_transposed, v_transposed, o_transposed} : (memref<5x4xf32>, memref<6x5xf32>, memref<7x6xf32>, memref<7x4xf32>) -> memref<7x4xf32>
  // CHECK: "test.get_gemm_gemm_size"(%op)
  "test.get_gemm_gemm_size"(%op) : (operation) -> !rock.gemm_gemm_size
  return
}

//===----------------------------------------------------------------------===//
// Test: 3D, only Q transposed
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @q_transposed_3d
func.func @q_transposed_3d(%q: memref<2x5x4xf32>, %k: memref<2x5x6xf32>, %v: memref<2x6x7xf32>, %out: memref<2x4x7xf32>) {
  %op = "rock.attention"(%q, %k, %v, %out) {q_transposed} : (memref<2x5x4xf32>, memref<2x5x6xf32>, memref<2x6x7xf32>, memref<2x4x7xf32>) -> memref<2x4x7xf32>
  // CHECK: "test.get_gemm_gemm_size"(%op)
  "test.get_gemm_gemm_size"(%op) : (operation) -> !rock.gemm_gemm_size
  return
}

//===----------------------------------------------------------------------===//
// Test: 3D, only K transposed
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @k_transposed_3d
func.func @k_transposed_3d(%q: memref<2x4x5xf32>, %k: memref<2x6x5xf32>, %v: memref<2x6x7xf32>, %out: memref<2x4x7xf32>) {
  %op = "rock.attention"(%q, %k, %v, %out) {k_transposed} : (memref<2x4x5xf32>, memref<2x6x5xf32>, memref<2x6x7xf32>, memref<2x4x7xf32>) -> memref<2x4x7xf32>
  // CHECK: "test.get_gemm_gemm_size"(%op)
  "test.get_gemm_gemm_size"(%op) : (operation) -> !rock.gemm_gemm_size
  return
}

//===----------------------------------------------------------------------===//
// Test: 3D, only V transposed
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @v_transposed_3d
func.func @v_transposed_3d(%q: memref<2x4x5xf32>, %k: memref<2x5x6xf32>, %v: memref<2x7x6xf32>, %out: memref<2x4x7xf32>) {
  %op = "rock.attention"(%q, %k, %v, %out) {v_transposed} : (memref<2x4x5xf32>, memref<2x5x6xf32>, memref<2x7x6xf32>, memref<2x4x7xf32>) -> memref<2x4x7xf32>
  // CHECK: "test.get_gemm_gemm_size"(%op)
  "test.get_gemm_gemm_size"(%op) : (operation) -> !rock.gemm_gemm_size
  return
}

//===----------------------------------------------------------------------===//
// Test: 3D, only O transposed
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @o_transposed_3d
func.func @o_transposed_3d(%q: memref<2x4x5xf32>, %k: memref<2x5x6xf32>, %v: memref<2x6x7xf32>, %out: memref<2x7x4xf32>) {
  %op = "rock.attention"(%q, %k, %v, %out) {o_transposed} : (memref<2x4x5xf32>, memref<2x5x6xf32>, memref<2x6x7xf32>, memref<2x7x4xf32>) -> memref<2x7x4xf32>
  // CHECK: "test.get_gemm_gemm_size"(%op)
  "test.get_gemm_gemm_size"(%op) : (operation) -> !rock.gemm_gemm_size
  return
}

//===----------------------------------------------------------------------===//
// Test: 3D, Q and O transposed
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @q_o_transposed_3d
func.func @q_o_transposed_3d(%q: memref<2x5x4xf32>, %k: memref<2x5x6xf32>, %v: memref<2x6x7xf32>, %out: memref<2x7x4xf32>) {
  %op = "rock.attention"(%q, %k, %v, %out) {q_transposed, o_transposed} : (memref<2x5x4xf32>, memref<2x5x6xf32>, memref<2x6x7xf32>, memref<2x7x4xf32>) -> memref<2x7x4xf32>
  // CHECK: "test.get_gemm_gemm_size"(%op)
  "test.get_gemm_gemm_size"(%op) : (operation) -> !rock.gemm_gemm_size
  return
}