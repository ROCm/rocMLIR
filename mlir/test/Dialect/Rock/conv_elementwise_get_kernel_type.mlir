// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: ConvElementwiseGemmOp::getKernelType always returns GemmElementwiseGemm
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @kernel_type_basic
func.func @kernel_type_basic(%a: memref<2x4xf32>, %b: memref<2x4xf32>, %c: memref<2x4xf32>, %out: memref<2x4xf32>) {
  %op = "rock.conv_elementwise_gemm"(%a, %b, %c, %out) : (memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>) -> memref<2x4xf32>
  // CHECK: "test.get_kernel_type"(%op)
  "test.get_kernel_type"(%op) : (operation) -> i32
  return
}

// CHECK-LABEL: func @kernel_type_transposed
func.func @kernel_type_transposed(%a: memref<2x4xf16>, %b: memref<2x4xf16>, %c: memref<2x4xf16>, %out: memref<2x4xf16>) {
  %op = "rock.conv_elementwise_gemm"(%a, %b, %c, %out) {a_transposed, b_transposed, c_transposed, o_transposed} : (memref<2x4xf16>, memref<2x4xf16>, memref<2x4xf16>, memref<2x4xf16>) -> memref<2x4xf16>
  // CHECK: "test.get_kernel_type"(%op)
  "test.get_kernel_type"(%op) : (operation) -> i32
  return
}

// CHECK-LABEL: func @kernel_type_batch
func.func @kernel_type_batch(%a: memref<3x4x5xf32>, %b: memref<3x5x6xf32>, %c: memref<3x6x7xf32>, %out: memref<3x4x7xf32>) {
  %op = "rock.conv_elementwise_gemm"(%a, %b, %c, %out) : (memref<3x4x5xf32>, memref<3x5x6xf32>, memref<3x6x7xf32>, memref<3x4x7xf32>) -> memref<3x4x7xf32>
  // CHECK: "test.get_kernel_type"(%op)
  "test.get_kernel_type"(%op) : (operation) -> i32
  return
}