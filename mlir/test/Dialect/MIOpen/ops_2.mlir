// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Run: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

func @miopen_gridwise_gemm_ex(%A : memref<?x?xf32>, %B : memref<?x?xf32>, %C : memref<?x?xf32>) {
  miopen.gridwise_gemm_ex(%A, %B, %C) {
    filter_layout = ["k", "c", "y", "x"],
    filter_dimension = [1, 2, 3, 4],
    input_layout = ["n", "c", "hi", "wi"],
    input_dimension = [5, 6, 7, 8],
    output_layout = ["n", "k", "ho", "wo"],
    output_dimension = [9, 10, 11, 12],
    strides = [1, 1],
    dilations = [1, 1],
    padding = [0, 0]
  } : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func @miopen_gridwise_gemm_ex
//  CHECK-NEXT: miopen.gridwise_gemm_ex

func @miopen_gpu_alloc() {
  %size = constant 1024 : index
  %c0 = constant 0 : index
  %c3 = constant 3 : index
  %c5 = constant 5 : index

  // allocation on global.
  %buffer_global = miopen.gpu_alloc(%size, %c0) : memref<1024xi8>

  // allocation on LDS.
  %buffer_lds = miopen.gpu_alloc(%size, %c3) : memref<1024xi8, 3>

  // allocation on register (VGPR).
  %buffer_register = miopen.gpu_alloc(%size, %c5) : memref<1024xi8, 5>

  return
}

// CHECK-LABEL: func @miopen_gpu_alloc
//   CHECK: miopen.gpu_alloc
//   CHECK-NEXT: miopen.gpu_alloc
//   CHECK-NEXT: miopen.gpu_alloc

func @miopen_subview(%buffer : memref<1024xi8>) {
  %c0 = constant 0 : index
  %c512 = constant 512 : index

  // 0 offset, same type.
  %view_0 = miopen.subview(%buffer, %c0) : memref<1024xi8> to memref<1024xi8>

  // 0 offset, different type.
  %view_1 = miopen.subview(%buffer, %c0) : memref<1024xi8> to memref<256xf32>

  // 0 offset, different type, different rank.
  %view_2 = miopen.subview(%buffer, %c0) { dimensions = [ 16, 16 ] } : memref<1024xi8> to memref<16x16xf32>

  // 512 offset, same type.
  %view_3 = miopen.subview(%buffer, %c512) : memref<1024xi8> to memref<512xi8>

  // 512 offset, different type.
  %view_4 = miopen.subview(%buffer, %c512) : memref<1024xi8> to memref<128xf32>

  // 512 offset, different type, different rank.
  %view_5 = miopen.subview(%buffer, %c512) { dimensions = [ 16, 8 ] } : memref<1024xi8> to memref<16x8xf32>

  return
}

// CHECK-LABEL: func @miopen_subview
//   CHECK: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview


func @miopen_fill(%buffer : memref<1024xf32, 5>) {
  %c0 = constant 0 : index
  miopen.fill(%buffer, %c0) : memref<1024xf32, 5>
  return
}

// CHECK-LABEL: func @miopen_fill
//   CHECK: miopen.fill

func @miopen_lds_barrier() {
  miopen.lds_barrier
  return
}

// CHECK-LABEL: func @miopen_lds_barrier
//   CHECK-NEXT: miopen.lds_barrier
 
func @miopen_blockwise_gemm(%A : memref<?x?xf32, 3>, %B : memref<?x?xf32, 3>, %C : memref<?x?xf32, 5>) {
  miopen.blockwise_gemm(%A, %B, %C) {
    m_per_thread = 64,
    n_per_thread = 64,
    k_per_thread = 16,

    m_level0_cluster = 16,
    n_level0_cluster = 16,
    m_level1_cluster = 16,
    n_level1_cluster = 16,

    matrix_a_source_data_per_read = 4,
    matrix_b_source_data_per_read = 4
  } : memref<?x?xf32, 3>, memref<?x?xf32, 3>, memref<?x?xf32, 5>
  return
}

// CHECK-LABEL: func @miopen_blockwise_gemm
//  CHECK-NEXT: miopen.blockwise_gemm

func @miopen_blockwise_copy(%source : memref<?x?xf32>, %dest : memref<?x?xf32, 3>) {
  miopen.blockwise_copy(%source, %dest) : memref<?x?xf32>, memref<?x?xf32, 3>
  miopen.blockwise_copy(%source, %dest) { move_source_offset = 16 } : memref<?x?xf32>, memref<?x?xf32, 3>
  return
}

// CHECK-LABEL: func @miopen_blockwise_copy
//  CHECK-NEXT: miopen.blockwise_copy
//  CHECK-NEXT: miopen.blockwise_copy

func @miopen_threadwise_copy(%source : memref<?x?xf32, 5>, %dest : memref<?x?xf32>) {
  miopen.threadwise_copy(%source, %dest) : memref<?x?xf32, 5>, memref<?x?xf32>
  miopen.threadwise_copy(%source, %dest) { offset_block = 16, offset_thread = 16 } : memref<?x?xf32, 5>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func @miopen_threadwise_copy
//  CHECK-NEXT: miopen.threadwise_copy
//  CHECK-NEXT: miopen.threadwise_copy
