// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: getFirstGemmIndex returns 0 for ConvElementwiseGemmOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @basic
func.func @basic(%input: memref<1x3x32x32xf32>, %filter: memref<1x3x3x3xf32>, %c: memref<1x3x30x30xf32>, %out: memref<1x3x30x30xf32>) {
  %op = "rock.conv_elementwise_gemm"(%input, %filter, %c, %out) : (memref<1x3x32x32xf32>, memref<1x3x3x3xf32>, memref<1x3x30x30xf32>, memref<1x3x30x30xf32>) -> memref<1x3x30x30xf32>
  // CHECK: "test.get_first_gemm_index"(%op)
  "test.get_first_gemm_index"(%op) : (operation) -> i32
  return
}

// CHECK-LABEL: func @with_transpose
func.func @with_transpose(%input: memref<2x4x16x16xf16>, %filter: memref<2x4x5x5xf16>, %c: memref<2x4x12x12xf16>, %out: memref<2x4x12x12xf16>) {
  %op = "rock.conv_elementwise_gemm"(%input, %filter, %c, %out) {a_transposed, b_transposed, c_transposed, o_transposed} : (memref<2x4x16x16xf16>, memref<2x4x5x5xf16>, memref<2x4x12x12xf16>, memref<2x4x12x12xf16>) -> memref<2x4x12x12xf16>
  // CHECK: "test.get_first_gemm_index"(%op)
  "test.get_first_gemm_index"(%op) : (operation) -> i32
  return
}

// CHECK-LABEL: func @batched
func.func @batched(%input: memref<8x3x32x32xf32>, %filter: memref<8x3x3x3xf32>, %c: memref<8x3x30x30xf32>, %out: memref<8x3x30x30xf32>) {
  %op = "rock.conv_elementwise_gemm"(%input, %filter, %c, %out) : (memref<8x3x32x32xf32>, memref<8x3x3x3xf32>, memref<8x3x30x30xf32>, memref<8x3x30x30xf32>) -> memref<8x3x30x30xf32>
  // CHECK: "test.get_first_gemm_index"(%op)
  "test.get_first_gemm_index"(%op) : (operation) -> i32
  return
}