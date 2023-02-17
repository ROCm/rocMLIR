// RUN: rocmlir-opt -rock-blockwise-gemm-to-threadwise %s | FileCheck %s

func.func @rock_blockwise_gemm_v2_two_results(%matrix : memref<1024xf32, 3>,
                                                %bufferA : memref<2xvector<2xf32>, 5>, %bufferB : memref<2xvector<2xf32>, 5>,
                                                %matrixC : memref<4xvector<16xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK:  rock.xdlops_gemm_v2
  rock.blockwise_gemm_v2 %matrixC += %bufferA from %matrix[%c0] * %bufferB from %matrix[%c0] {
    blockSize= 256 : i32,
    params = #rock.xdlops_gemm_params<
      kPerBlock = 2,
      kpack = 2,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      forceUnroll = true>,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 512 : index
  } : memref<4xvector<16xf32>, 5> += memref<2xvector<2xf32>, 5> from memref<1024xf32, 3> * memref<2xvector<2xf32>, 5> from memref<1024xf32, 3>
  return
}

func.func @rock_blockwise_gemm_v2_one_result(%matrix : memref<2048xi8, 3>,
                                               %bufferA : memref<1xvector<4xi8>, 5>, %bufferB : memref<1xvector<4xi8>, 5>,
                                               %matrixC : memref<1xvector<16xi32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK:  rock.xdlops_gemm_v2
  rock.blockwise_gemm_v2 %matrixC += %bufferA from %matrix[%c0] * %bufferB from %matrix[%c0] {
    blockSize = 256 : i32,
    params = #rock.xdlops_gemm_params<
      kPerBlock = 2,
      kpack = 8,
      mPerBlock = 64,
      mPerWave = 32,
      nPerBlock = 64,
      nPerWave = 32,
      forceUnroll = true>,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 1024 : index
  } : memref<1xvector<16xi32>, 5> += memref<1xvector<4xi8>, 5> from memref<2048xi8, 3> * memref<1xvector<4xi8>, 5> from memref<2048xi8, 3>
  return
}
