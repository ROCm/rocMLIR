// RUN: miopen-opt -miopen-threadwise-gemm-lowering %s | FileCheck %s

func.func @miopen_xdlops_gemm_v2_nonreduction_nokpack(%matrixA : memref<8xf32, 5>,
                                                      %matrixB : memref<8xf32, 5>,
                                                      %matrixC : memref<1xvector<32xf32>, 5>) {
  // CHECK: memref.load
  // CHECK: memref.load
  // CHECK: amdgpu.mfma
  miopen.xdlops_gemm_v2 %matrixC += %matrixA[0] * %matrixB[0] {
    params = #miopen.xdlops_gemm_params<
       kPerBlock = 8,
       kpack = 1,
       mPerBlock = 128,
       mPerWave = 64,
       nPerBlock = 64,
       nPerWave = 32>
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
    params = #miopen.xdlops_gemm_params<
      kPerBlock = 2,
      kpack = 2,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64>
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
    params = #miopen.xdlops_gemm_params<
      kPerBlock = 4,
      kpack = 8,
      mPerWave = 32,
      nPerWave = 32,
      mPerBlock = 64,
      nPerBlock = 64>
  } : memref<1xvector<16xi32>, 5> += memref<2xvector<8xi8>, 5> * memref<2xvector<8xi8>, 5>
  return
}
