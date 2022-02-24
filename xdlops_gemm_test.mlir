// RUN: miopen-opt -miopen-lowering-step4 %s | FileCheck %s

#map = affine_map<(d0) -> (d0 + 512)>

func @miopen_xdlops_gemm_v2_one_result(%matrixA : memref<1024xi8, 3>, %matrixB : memref<1024xi8, #map, 3>, %bufferA : memref<8xvector<16xi8>, 5>, %bufferB : memref<8xvector<16xi8>, 5>) -> vector<16xi32> {
  %c0 = arith.constant 0 : index
  %c0i = arith.constant 0 : i32
  %vectorC0 = vector.splat %c0i : vector<16xi32>
  // CHECK: miopen.mfma_v2
  // CHECK-NOT: miopen.mfma_v2
  %vectorD0 = miopen.xdlops_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    block_size = 256 : i32, // m_waves * n_waves * 64
    k = 16 : i32,
    kpack = 16 : i32,
    m_per_wave = 32 : i32, // xdlops requires 32x32
    n_per_wave = 32 : i32, // xdlops requires 32x32
    m = 64 : i32, // m_waves * m/wave
    n = 64 : i32, // n_waves * n/wave
    m_waves = 2 : i32,
    n_waves = 2 : i32,
    transforms = [[], []]
  } : memref<1024xi8, 3>, memref<1024xi8, #map, 3>, index, index, memref<8xvector<16xi8>, 5>, memref<8xvector<16xi8>, 5>, vector<16xi32> -> vector<16xi32>
  return %vectorD0 : vector<16xi32>
}
