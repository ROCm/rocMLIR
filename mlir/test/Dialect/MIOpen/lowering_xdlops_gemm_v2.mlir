// RUN: miopen-opt -miopen-threadwise-gemm-lowering %s | FileCheck %s

func.func @miopen_xdlops_gemm_v2_nonreduction_nokpack(%matrixA : memref<8xf32, 5>, 
                                                      %matrixB : memref<8xf32, 5>, 
                                                      %matrixC : memref<1xvector<32xf32>, 5>) {
  // CHECK: memref.load 
  // CHECK: memref.load 
  // CHECK: amdgpu.mfma
  miopen.xdlops_gemm_v2 %matrixC += %matrixA[0] * %matrixB[0] {
     k = 8 : i32, 
     kpack = 1 : i32, 
     m = 128 : i32, 
     m_per_wave = 64 : i32, 
     n = 64 : i32, 
     n_per_wave = 32 : i32
     } : memref<1xvector<32xf32>, 5> += memref<8xf32, 5> * memref<8xf32, 5>
  return
}

func.func @miopen_xdlops_gemm_v2_nonreduction_kpack(%matrixA : memref<2xvector<2xf32>, 5>, 
                                                    %matrixB : memref<2xvector<2xf32>, 5>,
                                                    %matrixC : memref<2xvector<32xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: miopen.extract_slice
  // CHECK: miopen.extract_slice
  // CHECK: amdgpu.mfma
  // CHECK: amdgpu.mfma
  miopen.xdlops_gemm_v2 %matrixC += %matrixA[0] * %matrixB[0] {
    block_size = 256 : i32,
    k = 2 : i32,
    kpack = 2 : i32,
    m = 128 : i32,
    m_per_wave = 64 : i32,
    m_waves = 2 : i32,
    n = 128 : i32,
    n_per_wave = 64 : i32,
    n_waves = 2 : i32
  } : memref<2xvector<32xf32>, 5> += memref<2xvector<2xf32>, 5> * memref<2xvector<2xf32>, 5>
  return
}

func.func @miopen_xdlops_gemm_v2_reduction_kpack(%matrixA : memref<2xvector<8xi8>, 5>, 
                                                 %matrixB : memref<2xvector<8xi8>, 5>, 
                                                 %matrixC : memref<1xvector<16xi32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: miopen.extract_slice
  // CHECK: miopen.extract_slice
  // CHECK: amdgpu.mfma
  // CHECK-NOT: amdgpu.mfma
  miopen.xdlops_gemm_v2 %matrixC += %matrixA[0] * %matrixB[0] {
    block_size = 256 : i32, // m_waves * n_waves * 64
    k = 4 : i32,
    kpack = 8 : i32,
    m_per_wave = 32 : i32, // xdlops requires 32x32
    n_per_wave = 32 : i32, // xdlops requires 32x32
    m = 64 : i32, // m_waves * m/wave
    n = 64 : i32, // n_waves * n/wave
    m_waves = 2 : i32,
    n_waves = 2 : i32
  } : memref<1xvector<16xi32>, 5> += memref<2xvector<8xi8>, 5> * memref<2xvector<8xi8>, 5>
  return
}
