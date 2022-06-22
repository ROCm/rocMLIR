// RUN: miopen-opt %s | FileCheck %s
// RUN: miopen-opt %s | miopen-opt | FileCheck %s
// Run: miopen-opt -mlir-print-op-generic %s | miopen-opt | FileCheck %s

func @miopen_threadwise_gemm_f16(%lhs : memref<32xf16, 5>, %rhs : memref<32xf16, 5>, %output : memref<64xf16, 5>) {
  miopen.threadwise_gemm %output += %lhs * %rhs
    { k = 4 : index, m = 8 : index, n = 8 : index, kPack = 1 : index }
    : memref<64xf16, 5> += memref<32xf16, 5> * memref<32xf16, 5>
  return
}

// CHECK-LABEL: func @miopen_threadwise_gemm_f16
//  CHECK: miopen.threadwise_gemm
