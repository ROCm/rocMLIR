// RUN: miopen-opt -miopen-lowering-step3 %s | FileCheck %s

#map = affine_map<(d0) -> (d0 + 512)>

func @miopen_blockwise_gemm_v2_two_results(%matrixA : memref<512xf32, 3>, %matrixB : memref<512xf32, #map, 3>, %bufferA : memref<2xvector<2xf32>, 5>, %bufferB : memref<2xvector<2xf32>, 5>) -> (vector<32xf32>, vector<32xf32>) {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f32
  %vectorC0 = splat %c0f : vector<32xf32>
  %vectorC1 = splat %c0f : vector<32xf32>
  // CHECK:  miopen.xdlops_gemm_v2
  %vectorD0, %vectorD1 = miopen.blockwise_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0, %vectorC1) {
    block_size = 256 : i32,
    k = 2 : i32,
    kpack = 2 : i32,
    m = 128 : i32,
    m_per_wave = 64 : i32,
    m_waves = 2 : i32,
    n = 128 : i32,
    n_per_wave = 64 : i32,
    n_waves = 2 : i32,
    transforms = [[], []]
  } : memref<512xf32, 3>, memref<512xf32, #map, 3>, index, index, memref<2xvector<2xf32>, 5>, memref<2xvector<2xf32>, 5>, vector<32xf32>, vector<32xf32> -> vector<32xf32>, vector<32xf32>
  return %vectorD0, %vectorD1 : vector<32xf32>, vector<32xf32>
}

func @miopen_blockwise_gemm_v2_one_result(%matrixA : memref<1024xi8, 3>, %matrixB : memref<1024xi8, #map, 3>, %bufferA : memref<2xvector<4xi8>, 5>, %bufferB : memref<2xvector<4xi8>, 5>) -> (vector<16xi32>) {
  %c0 = arith.constant 0 : index
  %c0i = arith.constant 0 : i32
  %vectorC0 = splat %c0i : vector<16xi32>
  // CHECK:  miopen.xdlops_gemm_v2
  %vectorD0 = miopen.blockwise_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    block_size = 256 : i32,
    k = 2 : i32,
    kpack = 8 : i32,
    m = 64 : i32,
    m_per_wave = 32 : i32,
    m_waves = 2 : i32,
    n = 64 : i32,
    n_per_wave = 32 : i32,
    n_waves = 2 : i32,
    transforms = [[], []]
  } : memref<1024xi8, 3>, memref<1024xi8, #map, 3>, index, index, memref<2xvector<4xi8>, 5>, memref<2xvector<4xi8>, 5>, vector<16xi32> -> vector<16xi32>
  return %vectorD0 : vector<16xi32>
}
