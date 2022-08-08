// RUN: miopen-opt -miopen-blockwise-gemm-to-threadwise %s | FileCheck %s

func.func @miopen_blockwise_gemm_v2_two_results(%matrix : memref<1024xf32, 3>, 
                                                %bufferA : memref<2xvector<2xf32>, 5>, %bufferB : memref<2xvector<2xf32>, 5>, 
                                                %matrixC : memref<2xvector<32xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK:  miopen.xdlops_gemm_v2
  miopen.blockwise_gemm_v2(%matrix, %matrix, %c0, %c0, %bufferA, %bufferB, %matrixC) {
    block_size = 256 : i32,
    k = 2 : i32,
    kpack = 2 : i32,
    m = 128 : i32,
    m_per_wave = 64 : i32,
    m_waves = 2 : i32,
    n = 128 : i32,
    n_per_wave = 64 : i32,
    n_waves = 2 : i32,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 512 : index
  } : memref<1024xf32, 3>, memref<1024xf32, 3>, index, index, memref<2xvector<2xf32>, 5>, memref<2xvector<2xf32>, 5>, memref<2xvector<32xf32>, 5>
  return
}

func.func @miopen_blockwise_gemm_v2_one_result(%matrix : memref<2048xi8, 3>, 
                                               %bufferA : memref<2xvector<4xi8>, 5>, %bufferB : memref<2xvector<4xi8>, 5>, 
                                               %matrixC : memref<1xvector<16xi32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK:  miopen.xdlops_gemm_v2
  miopen.blockwise_gemm_v2(%matrix, %matrix, %c0, %c0, %bufferA, %bufferB, %matrixC) {
    block_size = 256 : i32,
    k = 2 : i32,
    kpack = 8 : i32,
    m = 64 : i32,
    m_per_wave = 32 : i32,
    m_waves = 2 : i32,
    n = 64 : i32,
    n_per_wave = 32 : i32,
    n_waves = 2 : i32,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 1024 : index
  } : memref<2048xi8, 3>, memref<2048xi8, 3>, index, index, memref<2xvector<4xi8>, 5>, memref<2xvector<4xi8>, 5>, memref<1xvector<16xi32>, 5>
  return
}
