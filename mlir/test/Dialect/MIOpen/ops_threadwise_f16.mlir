// RUN: miopen-opt %s | FileCheck %s
// RUN: miopen-opt %s | miopen-opt | FileCheck %s
// Run: miopen-opt -mlir-print-op-generic %s | miopen-opt | FileCheck %s

func @miopen_threadwise_gemm_f16(%lhs : memref<4x8x1xf16, 5>, %rhs : memref<4x8x1xf16, 5>, %output : memref<8x8xf16, 5>) {
  miopen.threadwise_gemm %output += %lhs * %rhs
    : memref<8x8xf16, 5> += memref<4x8x1xf16, 5> * memref<4x8x1xf16, 5>
  return
}

// CHECK-LABEL: func @miopen_threadwise_gemm_f16
//  CHECK: miopen.threadwise_gemm
