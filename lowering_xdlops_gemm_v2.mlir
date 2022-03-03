// RUN: miopen-opt -miopen-lowering-step4 %s | FileCheck %s

#map = affine_map<(d0) -> (d0 + 512)>

func @miopen_xdlops_gemm_v2_two_results(%matrixA : memref<1024xi8, 3>, %matrixB : memref<1024xi8, #map, 3>, %bufferA : memref<2xvector<4xi8>, 5>, %bufferB : memref<2xvector<4xi8>, 5>) -> vector<16xi32> {
  %c0 = arith.constant 0 : index
  %c0i = arith.constant 0 : i32
  %vectorC0 = splat %c0i : vector<16xi32>
  //%vectorC1 = splat %c0i : vector<16xi32>
  // CHECK: miopen.mfma_v2
  // CHECK-NEXT: miopen.mfma_v2
  //%vectorD0 = miopen.xdlops_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0, %vectorC1) {
  %vectorD0 = miopen.xdlops_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    block_size = 256 : i32, // m_waves * n_waves * 64
    k = 2 : i32, // loop 2
    //kpack = 8 : i32, // loop another 2
    kpack = 16 : i32, // loop another 2
    m_per_wave = 32 : i32, // 32/16
    n_per_wave = 32 : i32, // 32/16
    m = 64 : i32, // m_waves * m/wave
    n = 64 : i32,
    m_waves = 2 : i32,
    n_waves = 2 : i32,
    transforms = [[], []]
  } : memref<1024xi8, 3>, memref<1024xi8, #map, 3>, index, index, memref<2xvector<4xi8>, 5>, memref<2xvector<4xi8>, 5>, vector<16xi32> -> vector<16xi32>
  return %vectorD0 : vector<16xi32>
}

// matrixA/B : <1024xi8, 3>
// m/wave n/wave 32
// m_waves n_waves 2
// block size 256
// m n 64
// k 2
// kpack 8
//  return %vectorD0, %vectorD1 : vector<16xi32>

