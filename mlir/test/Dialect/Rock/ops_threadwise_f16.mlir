// RUN: rocmlir-opt %s | FileCheck %s
// RUN: rocmlir-opt %s | rocmlir-opt | FileCheck %s
// Run: rocmlir-opt -mlir-print-op-generic %s | rocmlir-opt | FileCheck %s

// CHECK-LABEL: func.func @rock_threadwise_gemm_f16
//  CHECK: rock.threadwise_accel_gemm
func.func @rock_threadwise_gemm_f16(%lhs : memref<4x8xf16, 5>, %rhs : memref<4x8xf16, 5>, %output : memref<8x8xf16, 5>) {
  %c1 = arith.constant 1 : index
  rock.threadwise_accel_gemm %output += %lhs * %rhs at [%c1, %c1, %c1] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.wmma_gemm_params<
       mPerBlock = 32,
       nPerBlock = 32,
       kpackPerBlock = 2,
       mPerWave = 32,
       nPerWave = 16,
       kpack = 16,
       splitKFactor = 3, 
       scheduleVersion = 1, 
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<8x8xf16, 5> += memref<4x8xf16, 5> * memref<4x8xf16, 5>
  return
}
