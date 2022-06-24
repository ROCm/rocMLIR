// RUN: miopen-opt -miopen-lowering-step4 %s | FileCheck %s

func @miopen_xdlops_gemm_v2_two_results(%matrix : memref<1024xf32, 3>, %bufferA : memref<2xvector<2xf32>, 5>, %bufferB : memref<2xvector<2xf32>, 5>) -> (vector<32xf32>, vector<32xf32>) {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f32
  %vectorC0 = vector.splat %c0f : vector<32xf32>
  %vectorC1 = vector.splat %c0f : vector<32xf32>
  // CHECK: amdgpu.mfma
  // CHECK-NEXT: amdgpu.mfma
  %vectorD0, %vectorD1 = miopen.xdlops_gemm_v2(%matrix, %matrix, %c0, %c0, %bufferA, %bufferB, %vectorC0, %vectorC1) {
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
  } : memref<1024xf32, 3>, memref<1024xf32, 3>, index, index, memref<2xvector<2xf32>, 5>, memref<2xvector<2xf32>, 5>, vector<32xf32>, vector<32xf32> -> vector<32xf32>, vector<32xf32>
  return %vectorD0, %vectorD1 : vector<32xf32>, vector<32xf32>
}

func @miopen_xdlops_gemm_v2_one_result(%matrix : memref<2048xi8, 3>, %bufferA : memref<2xvector<4xi8>, 5>, %bufferB : memref<2xvector<4xi8>, 5>) -> vector<16xi32> {
  %c0 = arith.constant 0 : index
  %c0i = arith.constant 0 : i32
  %vectorC0 = vector.splat %c0i : vector<16xi32>
  // CHECK: amdgpu.mfma
  // CHECK-NOT: amdgpu.mfma
  %vectorD0 = miopen.xdlops_gemm_v2(%matrix, %matrix, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    block_size = 256 : i32, // m_waves * n_waves * 64
    k = 2 : i32,
    kpack = 8 : i32,
    m_per_wave = 32 : i32, // xdlops requires 32x32
    n_per_wave = 32 : i32, // xdlops requires 32x32
    m = 64 : i32, // m_waves * m/wave
    n = 64 : i32, // n_waves * n/wave
    m_waves = 2 : i32,
    n_waves = 2 : i32,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 1024 : index
  } : memref<2048xi8, 3>, memref<2048xi8, 3>, index, index, memref<2xvector<4xi8>, 5>, memref<2xvector<4xi8>, 5>, vector<16xi32> -> vector<16xi32>
  return %vectorD0 : vector<16xi32>
}
