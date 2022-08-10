// RUN: miopen-opt %s | FileCheck %s
// RUN: miopen-opt %s | miopen-opt | FileCheck %s
// Run: miopen-opt -mlir-print-op-generic %s | miopen-opt | FileCheck %s

func.func @miopen_blockwise_gemm_f16(%A : memref<8x128x1xf16, 3>, %B : memref<8x128x1xf16, 3>, %C : memref<8x8xf16, 5>) {
  miopen.blockwise_gemm %C += %A * %B {
    kPerThread = 1 : index,
    mPerThread = 4 : index,
    mThreadsPerCuwave = 4 : index,
    mCuwavesPerBlock = 4 : index,
    nPerThread = 4 : index,
    nThreadsPerCuwave = 4 : index,
    nCuwavesPerBlock = 4 : index
  } : memref<8x8xf16, 5> += memref<8x128x1xf16, 3> * memref<8x128x1xf16, 3>
  return
}

// CHECK-LABEL: func.func @miopen_blockwise_gemm_f16
//  CHECK: miopen.blockwise_gemm

// ----

func.func @miopen_xdlops_gemm_v2_one_result_f16(%matrixA : memref<32xf16, 5>, 
                                                %matrixB : memref<16xf16, 5>, 
                                                %matrixC : memref<1xvector<32xf16>, 5>) {
  %c0 = arith.constant 0 : index
  miopen.xdlops_gemm_v2 %matrixC += %matrixA[0] * %matrixB[0] {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64
  } : memref<1xvector<32xf16>, 5> += memref<32xf16, 5> * memref<16xf16, 5>
  return
}

// CHECK-LABEL: func.func @miopen_xdlops_gemm_v2_one_result_f16
//  CHECK: miopen.xdlops_gemm_v2

// ----

func.func @miopen_xdlops_gemm_v2_two_results_f16(%matrixA : memref<32xf16, 5>, 
                                                 %matrixB: memref<16xf16, 5>,
                                                 %matrixC : memref<1xvector<32xf16>, 5>) {
  %c0 = arith.constant 0 : index
  miopen.xdlops_gemm_v2 %matrixC += %matrixA[0] * %matrixB[0] {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64
  } : memref<1xvector<32xf16>, 5> += memref<32xf16, 5> * memref<16xf16, 5>
  return
}

// CHECK-LABEL: func.func @miopen_xdlops_gemm_v2_two_results_f16
//  CHECK: miopen.xdlops_gemm_v2

// ----

func.func @miopen_blockwise_gemm_v2_one_result_f16(%matrix : memref<12288xf16, 3>,
                                          %bufferA : memref<32xf16, 5>, %bufferB : memref<16xf16, 5>, 
                                          %matrixC : memref<1xvector<32xf16>, 5>) {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f16
  miopen.blockwise_gemm_v2 %matrixC += %bufferA from %matrix[%c0] * %bufferB from %matrix[%c0] {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 8192 : index
  } : memref<1xvector<32xf16>, 5> +=  memref<32xf16, 5> from memref<12288xf16, 3> * memref<16xf16, 5> from memref<12288xf16, 3>
  return
}

// CHECK-LABEL: func.func @miopen_blockwise_gemm_v2_one_result_f16
//  CHECK: miopen.blockwise_gemm_v2

// ----

func.func @miopen_blockwise_gemm_v2_two_results_f16(%matrix : memref<12288xf16, 3>,
                                               %bufferA : memref<32xf16, 5>, %bufferB : memref<16xf16, 5>,
                                               %matrixC : memref<2xvector<32xf16>, 5>) {
  %c0 = arith.constant 0 : index
  miopen.blockwise_gemm_v2 %matrixC += %bufferA from %matrix[%c0] * %bufferB from %matrix[%c0] {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 8192 : index
  } : memref<2xvector<32xf16>, 5> += memref<32xf16, 5> from memref<12288xf16, 3> * memref<16xf16, 5> from memref<12288xf16, 3>
  return
}

// CHECK-LABEL: func.func @miopen_blockwise_gemm_v2_two_results_f16
//  CHECK: miopen.blockwise_gemm_v2
