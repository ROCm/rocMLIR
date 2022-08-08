// RUN: miopen-opt -miopen-threadwise-gemm-lowering %s | FileCheck %s

func.func @miopen_xdlops_gemm_v2_nonreduction_nokpack(%matrix : memref<1536xf32, 3>, 
                                                      %bufferA : memref<8xf32, 5>, %bufferB : memref<8xf32, 5>, 
                                                      %matrixC : memref<1xvector<32xf32>, 5>) {
  // CHECK: memref.load 
  // CHECK: memref.load 
  // CHECK: amdgpu.mfma
  miopen.xdlops_gemm_v2(%matrix, %matrix, %bufferA, %bufferB, %matrixC) {
     k = 8 : i32, 
     kpack = 1 : i32, 
     ldsBufferOffsetA = 0 : index, 
     ldsBufferOffsetB = 1024 : index, 
     m = 128 : i32, 
     m_per_wave = 64 : i32, 
     n = 64 : i32, 
     n_per_wave = 32 : i32,
     regOffsetA = 0 : index,
     regOffsetB = 0 : index
     } : memref<1536xf32, 3>, memref<1536xf32, 3>, memref<8xf32, 5>, memref<8xf32, 5>, memref<1xvector<32xf32>, 5>
  return
}

func.func @miopen_xdlops_gemm_v2_nonreduction_kpack(%matrix : memref<1024xf32, 3>, 
                                                    %bufferA : memref<2xvector<2xf32>, 5>, %bufferB : memref<2xvector<2xf32>, 5>,
                                                    %matrixC : memref<2xvector<32xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: miopen.extract_slice
  // CHECK: miopen.extract_slice
  // CHECK: amdgpu.mfma
  // CHECK: amdgpu.mfma
  miopen.xdlops_gemm_v2(%matrix, %matrix, %bufferA, %bufferB, %matrixC) {
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
    ldsBufferOffsetB = 512 : index,
    regOffsetA = 0 : index,
    regOffsetB = 0 : index
  } : memref<1024xf32, 3>, memref<1024xf32, 3>, memref<2xvector<2xf32>, 5>, memref<2xvector<2xf32>, 5>, memref<2xvector<32xf32>, 5>
  return
}

func.func @miopen_xdlops_gemm_v2_reduction_kpack(%matrix : memref<2048xi8, 3>, 
                                                 %bufferA : memref<2xvector<8xi8>, 5>, %bufferB : memref<2xvector<8xi8>, 5>, 
                                                 %matrixC : memref<1xvector<16xi32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: miopen.extract_slice
  // CHECK: miopen.extract_slice
  // CHECK: amdgpu.mfma
  // CHECK-NOT: amdgpu.mfma
  miopen.xdlops_gemm_v2(%matrix, %matrix, %bufferA, %bufferB, %matrixC) {
    block_size = 256 : i32, // m_waves * n_waves * 64
    k = 4 : i32,
    kpack = 8 : i32,
    m_per_wave = 32 : i32, // xdlops requires 32x32
    n_per_wave = 32 : i32, // xdlops requires 32x32
    m = 64 : i32, // m_waves * m/wave
    n = 64 : i32, // n_waves * n/wave
    m_waves = 2 : i32,
    n_waves = 2 : i32,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 1024 : index,
    regOffsetA = 0 : index,
    regOffsetB = 0 : index
  } : memref<2048xi8, 3>, memref<2048xi8, 3>, memref<2xvector<8xi8>, 5>, memref<2xvector<8xi8>, 5>, memref<1xvector<16xi32>, 5> 
  return
}
