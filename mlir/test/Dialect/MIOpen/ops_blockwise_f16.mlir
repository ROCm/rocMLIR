// RUN: miopen-opt %s | FileCheck %s
// RUN: miopen-opt %s | miopen-opt | FileCheck %s
// Run: miopen-opt -mlir-print-op-generic %s | miopen-opt | FileCheck %s

func @miopen_blockwise_gemm_f16(%A : memref<?x?x?xf16, 3>, %B : memref<?x?x?xf16, 3>, %C : memref<?x?x?xf16, 5>) {
  %c0 = constant 0 : index
  miopen.blockwise_gemm(%A, %B, %C, %c0, %c0) {
    m_per_thread = 64,
    n_per_thread = 64,
    k_per_thread = 16,

    m_level0_cluster = 16,
    n_level0_cluster = 16,
    m_level1_cluster = 16,
    n_level1_cluster = 16,

    matrix_a_source_data_per_read = 4,
    matrix_b_source_data_per_read = 4
  } : memref<?x?x?xf16, 3>, memref<?x?x?xf16, 3>, memref<?x?x?xf16, 5>, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_gemm_f16
//  CHECK: miopen.blockwise_gemm

func @miopen_blockwise_copy_f16(%source : memref<?x?x?xf16>, %dest : memref<?x?x?xf16, 3>, %sc0 : i32, %sc1 : i32, %sc2 : i32, %dc0 : i32, %dc1 : i32, %dc2 : i32) {
  miopen.blockwise_copy %source[%sc0, %sc1, %sc2] -> %dest[%dc0, %dc1, %dc2] : memref<?x?x?xf16>, i32, i32, i32 -> memref<?x?x?xf16, 3>, i32, i32, i32
  miopen.blockwise_copy %source[%sc0, %sc1, %sc2] -> %dest[%dc0, %dc1, %dc2] { move_source_offset = 16 } : memref<?x?x?xf16>, i32, i32, i32 -> memref<?x?x?xf16, 3>, i32, i32, i32
  return
}

// CHECK-LABEL: func @miopen_blockwise_copy_f16
//  CHECK-NEXT: miopen.blockwise_copy
//  CHECK-NEXT: miopen.blockwise_copy

// ----

func @miopen_xdlops_gemm_v2_one_result_f16(%matrixA : memref<12288xf16, 3>, %matrixB : memref<12288xf16, 3>,
                                       %bufferA : memref<32xf16, 5>, %bufferB : memref<16xf16, 5>) -> vector<32xf16> {
  %c0 = constant 0 : index
  %c0f = constant 0.0 : f16
  %vectorC0 = splat %c0f : vector<32xf16>
  %vectorD0 = miopen.xdlops_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf16, 3>, memref<12288xf16, 3>, index, index, memref<32xf16, 5>, memref<16xf16, 5>, vector<32xf16> -> vector<32xf16>
  return %vectorD0 : vector<32xf16>
}

// CHECK-LABEL: func @miopen_xdlops_gemm_v2_one_result_f16
//  CHECK: miopen.xdlops_gemm_v2

// ----

func @miopen_xdlops_gemm_v2_two_results_f16(%matrixA : memref<12288xf16, 3>, %matrixB : memref<12288xf16, 3>,
                                        %bufferA : memref<32xf16, 5>, %bufferB: memref<16xf16, 5>) -> (vector<32xf16>, vector<32xf16>) {
  %c0 = constant 0 : index
  %c0f = constant 0.0 : f16
  %vectorC0 = splat %c0f : vector<32xf16>
  %vectorC1 = splat %c0f : vector<32xf16>
  %vectorD0, %vectorD1 = miopen.xdlops_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0, %vectorC1) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf16, 3>, memref<12288xf16, 3>, index, index, memref<32xf16, 5>, memref<16xf16, 5>, vector<32xf16>, vector<32xf16> -> vector<32xf16>, vector<32xf16>
  return %vectorD0, %vectorD1 : vector<32xf16>, vector<32xf16>
}

// CHECK-LABEL: func @miopen_xdlops_gemm_v2_two_results_f16
//  CHECK: miopen.xdlops_gemm_v2

// ----

func @miopen_blockwise_gemm_v2_one_result_f16(%matrixA : memref<12288xf16, 3>, %matrixB : memref<12288xf16, 3>,
                                          %bufferA : memref<32xf16, 5>, %bufferB : memref<16xf16, 5>) -> vector<32xf16> {
  %c0 = constant 0 : index
  %c0f = constant 0.0 : f16
  %vectorC0 = splat %c0f : vector<32xf16>
  %vectorD0 = miopen.blockwise_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf16, 3>, memref<12288xf16, 3>, index, index, memref<32xf16, 5>, memref<16xf16, 5>, vector<32xf16> -> vector<32xf16>
  return %vectorD0 : vector<32xf16>
}

// CHECK-LABEL: func @miopen_blockwise_gemm_v2_one_result_f16
//  CHECK: miopen.blockwise_gemm_v2

// ----

func @miopen_blockwise_gemm_v2_two_results_f16(%matrixA : memref<12288xf16, 3>, %matrixB : memref<12288xf16, 3>,
                                           %bufferA : memref<32xf16, 5>, %bufferB : memref<16xf16, 5>) -> (vector<32xf16>, vector<32xf16>) {
  %c0 = constant 0 : index
  %c0f = constant 0.0 : f16
  %vectorC0 = splat %c0f : vector<32xf16>
  %vectorC1 = splat %c0f : vector<32xf16>
  %vectorD0, %vectorD1 = miopen.blockwise_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0, %vectorC1) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf16, 3>, memref<12288xf16, 3>, index, index, memref<32xf16, 5>, memref<16xf16, 5>, vector<32xf16>, vector<32xf16> -> vector<32xf16>, vector<32xf16>
  return %vectorD0, %vectorD1 : vector<32xf16>, vector<32xf16>
}

// CHECK-LABEL: func @miopen_blockwise_gemm_v2_two_results_f16
//  CHECK: miopen.blockwise_gemm_v2
