// RUN: rocmlir-opt %s | FileCheck %s
// RUN: rocmlir-opt %s | rocmlir-opt | FileCheck %s
// Run: rocmlir-opt -mlir-print-op-generic %s | rocmlir-opt | FileCheck %s

func.func @rock_threadwise_gemm_f16(%lhs : memref<4x8x1xf16, 5>, %rhs : memref<4x8x1xf16, 5>, %output : memref<8x8xf16, 5>) {
  rock.threadwise_gemm %output += %lhs * %rhs
    : memref<8x8xf16, 5> += memref<4x8x1xf16, 5> * memref<4x8x1xf16, 5>
  return
}

// CHECK-LABEL: func.func @rock_threadwise_gemm_f16
//  CHECK: rock.threadwise_gemm
