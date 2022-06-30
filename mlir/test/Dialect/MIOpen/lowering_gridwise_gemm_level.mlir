// This tests checks the following aspects of lowering component:
// * Can pass arguments correctly
// * Can pass arguments in the right sequence
// * Have the right number of transforms
// * Have one gridwise_gemm

// RUN: miopen-opt -miopen-conv-to-gemm %s | FileCheck %s

func @miopen_gridwise_gemm(%matrix_a : memref<?x?x?xf32>, %matrix_b : memref<?x?x?xf32>, %matrix_c : memref<?x?x?xf32>) {
  miopen.gridwise_gemm(%matrix_a, %matrix_b, %matrix_c) {
    transforms = [[], [], []],
    paddingInfo = #miopen.padding_info<extraM = 0, extraN = 0, extraK = 0>,

    block_size = 256,

    m_per_block = 128,
    n_per_block = 128,
    k_per_block = 16,

    m_per_thread = 64,
    n_per_thread = 64,
    k_per_thread = 16,

    m_level0_cluster = 16,
    n_level0_cluster = 16,
    m_level1_cluster = 16,
    n_level1_cluster = 16,

    matrix_a_source_vector_read_dim = 0,
    matrix_a_source_data_per_read = 4,
    matrix_a_dest_data_per_write_dim_m = 4,

    matrix_b_source_vector_read_dim = 1,
    matrix_b_source_data_per_read = 4,
    matrix_b_dest_data_per_write_dim_n = 4,

    matrix_c_source_dest_vector_read_write_dim = 3,
    matrix_c_dest_data_per_write = 1
  } : memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>
  return
}

// TBD: add lowering checks
// CHECK-LABEL: func @miopen_gridwise_gemm{{.*}}%arg0{{.*}}%arg1{{.*}}%arg2
