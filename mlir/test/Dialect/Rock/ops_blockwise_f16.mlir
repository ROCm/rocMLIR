// RUN: rocmlir-opt %s | FileCheck %s
// RUN: rocmlir-opt %s | rocmlir-opt | FileCheck %s
// Run: rocmlir-opt -mlir-print-op-generic %s | rocmlir-opt | FileCheck %s

func.func @rock_blockwise_gemm_f16(%A : memref<8x128x1xf16, 3>, %B : memref<8x128x1xf16, 3>, %C : memref<8x8xf16, 5>) {
  rock.blockwise_gemm %C += %A * %B {
    params = #rock.general_gemm_params<
      blockSize = 256,
      kPerBlock = 8,
      mPerBlock = 256,
      nPerBlock = 256,
      kPerThread = 1,
      mPerThread = 4,
      nPerThread = 4,
      kpack = 1>
  } : memref<8x8xf16, 5> += memref<8x128x1xf16, 3> * memref<8x128x1xf16, 3>
  return
}

// CHECK-LABEL: func.func @rock_blockwise_gemm_f16
//  CHECK: rock.blockwise_gemm

// ----

func.func @rock_xdlops_gemm_v2_one_result_f16(%matrixA : memref<4xvector<4xf16>, 5>,
                                                %matrixB : memref<4xvector<4xf16>, 5>,
                                                %matrixC : memref<1xvector<32xf32>, 5>) {
  %c0 = arith.constant 0 : index
  rock.accel_gemm_v2 %matrixC += %matrixA[%c0] * %matrixB[%c0] {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    params = #rock.accel_gemm_params<
      mPerBlock = 256,
      nPerBlock = 256,
      kPerBlock = 16,
      mPerWave = 128,
      nPerWave = 64,
      kpack = 1,
      forceUnroll = true>
  } : memref<1xvector<32xf32>, 5> += memref<4xvector<4xf16>, 5> * memref<4xvector<4xf16>, 5>
  return
}

// CHECK-LABEL: func.func @rock_xdlops_gemm_v2_one_result_f16
//  CHECK: rock.accel_gemm_v2

// ----

func.func @rock_xdlops_gemm_v2_two_results_f16(%matrixA : memref<4xvector<4xf16>, 5>,
                                               %matrixB : memref<4xvector<4xf16>, 5>,
                                               %matrixC : memref<1xvector<32xf32>, 5>) {
  %c0 = arith.constant 0 : index
  rock.accel_gemm_v2 %matrixC += %matrixA[%c0] * %matrixB[%c0] {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    params = #rock.accel_gemm_params<
      mPerBlock = 256,
      nPerBlock = 256,
      kPerBlock = 16,
      mPerWave = 128,
      nPerWave = 64,
      kpack = 1,
      forceUnroll = true>
  } : memref<1xvector<32xf32>, 5> += memref<4xvector<4xf16>, 5> * memref<4xvector<4xf16>, 5>
  return
}

// CHECK-LABEL: func.func @rock_xdlops_gemm_v2_two_results_f16
//  CHECK: rock.accel_gemm_v2

// ----

func.func @rock_blockwise_gemm_v2_one_result_f16(%matrixA : memref<8192xf16, 3>, %matrixB : memref<4096xf16, 3>,
                                          %bufferA : memref<4xvector<4xf16>, 5>, %bufferB : memref<4xvector<4xf16>, 5>,
                                          %matrixC : memref<1xvector<32xf32>, 5>) {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f16
  rock.blockwise_gemm_v2 %matrixC += %bufferA from %matrixA[%c0] * %bufferB from %matrixB[%c0] {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    blockSize = 256 : i32,
    params = #rock.accel_gemm_params<
      mPerBlock = 256,
      nPerBlock = 256,
      kPerBlock = 16,
      mPerWave = 16,
      nPerWave = 16,
      kpack = 1,
      forceUnroll = true>
  } : memref<1xvector<32xf32>, 5> +=  memref<4xvector<4xf16>, 5> from memref<8192xf16, 3> * memref<4xvector<4xf16>, 5> from memref<4096xf16, 3>
  return
}

// CHECK-LABEL: func.func @rock_blockwise_gemm_v2_one_result_f16
//  CHECK: rock.blockwise_gemm_v2

// ----

func.func @rock_blockwise_gemm_v2_two_results_f16(%matrixA : memref<8192xf16, 3>, %matrixB : memref<4096xf16, 3>,
                                               %bufferA : memref<4xvector<4xf16>, 5>, %bufferB : memref<4xvector<4xf16>, 5>,
                                               %matrixC : memref<2xvector<32xf32>, 5>) {
  %c0 = arith.constant 0 : index
  rock.blockwise_gemm_v2 %matrixC += %bufferA from %matrixA[%c0] * %bufferB from %matrixB[%c0] {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    blockSize = 256 : i32,
    params = #rock.accel_gemm_params<
      mPerBlock = 256,
      nPerBlock = 256,
      kPerBlock = 16,
      mPerWave = 128,
      nPerWave = 64,
      kpack = 1,
      forceUnroll = true>
  } : memref<2xvector<32xf32>, 5> += memref<4xvector<4xf16>, 5> from memref<8192xf16, 3> * memref<4xvector<4xf16>, 5> from memref<4096xf16, 3>
  return
}

// CHECK-LABEL: func.func @rock_blockwise_gemm_v2_two_results_f16
//  CHECK: rock.blockwise_gemm_v2
