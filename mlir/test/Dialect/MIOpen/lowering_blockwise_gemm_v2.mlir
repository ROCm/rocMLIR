// RUN: miopen-opt -miopen-lowering-step3 %s | FileCheck %s

func @miopen_blockwise_gemm_v2_two_results(%matrixA : memref<12288xf32, 3>, %matrixB : memref<12288xf32, 3>, %bufferA : memref<4xf32, 5>, %bufferB : memref<4xf32, 5>) -> (vector<32xf32>, vector<32xf32>) {
  %c0 = constant 0 : index
  %c0f = constant 0.0 : f32
  %vectorC0 = splat %c0f : vector<32xf32>
  %vectorC1 = splat %c0f : vector<32xf32>
  // CHECK:  miopen.xdlops_gemm_v2
  %vectorD0, %vectorD1 = miopen.blockwise_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0, %vectorC1) {
    m = 256,
    n = 256,
    k = 16,
    kpack = 1,
    block_size = 256,
    m_per_wave = 64,
    n_per_wave = 64,
    m_waves = 4,
    n_waves = 4,
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, index, index, memref<4xf32, 5>, memref<4xf32, 5>, vector<32xf32>, vector<32xf32> -> vector<32xf32>, vector<32xf32>
  return %vectorD0, %vectorD1 : vector<32xf32>, vector<32xf32>
}
