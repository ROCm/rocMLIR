// RUN: miopen-opt %s | FileCheck %s
// RUN: miopen-opt %s | miopen-opt | FileCheck %s
// Run: miopen-opt -mlir-print-op-generic %s | miopen-opt | FileCheck %s

func @miopen_blockwise_gemm_f16(%A : memref<8x128x1xf16, 3>, %B : memref<8x128x1xf16, 3>, %C : memref<8x8xf16, 5>) {
  %c0 = arith.constant 0 : index
  miopen.blockwise_gemm(%A, %B, %C, %c0, %c0) {
    kPerThread = 1 : index,
    mPerThread = 4 : index,
    mRepeatStride = 64 : index,
    nPerThread = 4 : index,
    nRepeatStride = 64 : index
  } : memref<8x128x1xf16, 3>, memref<8x128x1xf16, 3>, memref<8x8xf16, 5>, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_gemm_f16
//  CHECK: miopen.blockwise_gemm

// ----

func @miopen_xdlops_gemm_v2_one_result_f16(%matrix : memref<12288xf16, 3>,
                                       %bufferA : memref<32xf16, 5>, %bufferB : memref<16xf16, 5>) -> vector<32xf16> {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f16
  %vectorC0 = vector.splat %c0f : vector<32xf16>
  %vectorD0 = miopen.xdlops_gemm_v2(%matrix, %matrix, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 8192 : index
  } : memref<12288xf16, 3>, memref<12288xf16, 3>, index, index, memref<32xf16, 5>, memref<16xf16, 5>, vector<32xf16> -> vector<32xf16>
  return %vectorD0 : vector<32xf16>
}

// CHECK-LABEL: func @miopen_xdlops_gemm_v2_one_result_f16
//  CHECK: miopen.xdlops_gemm_v2

// ----

func @miopen_xdlops_gemm_v2_two_results_f16(%matrix : memref<12288xf16, 3>,
                                        %bufferA : memref<32xf16, 5>, %bufferB: memref<16xf16, 5>) -> (vector<32xf16>, vector<32xf16>) {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f16
  %vectorC0 = vector.splat %c0f : vector<32xf16>
  %vectorC1 = vector.splat %c0f : vector<32xf16>
  %vectorD0, %vectorD1 = miopen.xdlops_gemm_v2(%matrix, %matrix, %c0, %c0, %bufferA, %bufferB, %vectorC0, %vectorC1) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 8192 : index
  } : memref<12288xf16, 3>, memref<12288xf16, 3>, index, index, memref<32xf16, 5>, memref<16xf16, 5>, vector<32xf16>, vector<32xf16> -> vector<32xf16>, vector<32xf16>
  return %vectorD0, %vectorD1 : vector<32xf16>, vector<32xf16>
}

// CHECK-LABEL: func @miopen_xdlops_gemm_v2_two_results_f16
//  CHECK: miopen.xdlops_gemm_v2

// ----

func @miopen_blockwise_gemm_v2_one_result_f16(%matrix : memref<12288xf16, 3>,
                                          %bufferA : memref<32xf16, 5>, %bufferB : memref<16xf16, 5>) -> vector<32xf16> {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f16
  %vectorC0 = vector.splat %c0f : vector<32xf16>
  %vectorD0 = miopen.blockwise_gemm_v2(%matrix, %matrix, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 8192 : index
  } : memref<12288xf16, 3>, memref<12288xf16, 3>, index, index, memref<32xf16, 5>, memref<16xf16, 5>, vector<32xf16> -> vector<32xf16>
  return %vectorD0 : vector<32xf16>
}

// CHECK-LABEL: func @miopen_blockwise_gemm_v2_one_result_f16
//  CHECK: miopen.blockwise_gemm_v2

// ----

func @miopen_blockwise_gemm_v2_two_results_f16(%matrix : memref<12288xf16, 3>,
                                           %bufferA : memref<32xf16, 5>, %bufferB : memref<16xf16, 5>) -> (vector<32xf16>, vector<32xf16>) {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f16
  %vectorC0 = vector.splat %c0f : vector<32xf16>
  %vectorC1 = vector.splat %c0f : vector<32xf16>
  %vectorD0, %vectorD1 = miopen.blockwise_gemm_v2(%matrix, %matrix, %c0, %c0, %bufferA, %bufferB, %vectorC0, %vectorC1) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 8192 : index
  } : memref<12288xf16, 3>, memref<12288xf16, 3>, index, index, memref<32xf16, 5>, memref<16xf16, 5>, vector<32xf16>, vector<32xf16> -> vector<32xf16>, vector<32xf16>
  return %vectorD0, %vectorD1 : vector<32xf16>, vector<32xf16>
}

// CHECK-LABEL: func @miopen_blockwise_gemm_v2_two_results_f16
//  CHECK: miopen.blockwise_gemm_v2
