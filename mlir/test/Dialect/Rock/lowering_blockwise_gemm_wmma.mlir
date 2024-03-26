// RUN: rocmlir-opt -rock-blockwise-gemm-to-threadwise %s | FileCheck %s
#wg = #gpu.address_space<workgroup>
#priv = #gpu.address_space<private>

func.func @rock_blockwise_gemm_accel_wmma(%matrixA : memref<16xvector<8xf16>, #wg>, %matrixB : memref<16xvector<8xf16>, #wg>,
                                          %bufferA : memref<1xvector<16xf16>, #priv>, %bufferB : memref<1xvector<16xf16>, #priv>,
                                          %matrixC : memref<1xvector<8xf32>, #priv>) {
  // CHECK: affine.for {{.*}} = 0 to 1
  // CHECK: rock.threadwise_read_into
  // CHECK: affine.for {{.*}} = 0 to 1
  // CHECK: rock.threadwise_read_into
  // CHECK: rock.threadwise_accel_gemm
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA * %bufferB from %matrixB features = wmma{
    arch = "amdgcn-amd-amdhsa:gfx1100",
    blockSize = 32 : i32,
    inMPerThread = 2 : i32,
    inNPerThread = 2 : i32,
    params = #rock.wmma_gemm_params<
      kpackPerBlock = 4,
      kpack = 8,
      mPerBlock = 16,
      mPerWave = 16,
      nPerBlock = 16,
      nPerWave = 16,
      splitKFactor = 1,
      forceUnroll = true>
  } : memref<1xvector<8xf32>, #priv> += memref<1xvector<16xf16>, #priv> from memref<16xvector<8xf16>, #wg> * memref<1xvector<16xf16>, #priv> from memref<16xvector<8xf16>, #wg>
  return
}

func.func @rock_blockwise_gemm_accel_wmma_largekpack(%matrixA : memref<32xvector<8xf16>, #wg>, %matrixB : memref<32xvector<8xf16>, #wg>,
                                                     %bufferA : memref<1xvector<16xf16>, #priv>, %bufferB : memref<1xvector<16xf16>, #priv>,
                                                     %matrixC : memref<1xvector<8xf32>, #priv>) {
  // CHECK: affine.for {{.*}} = 0 to 1
  // CHECK: rock.threadwise_read_into
  // CHECK: affine.for {{.*}} = 0 to 1
  // CHECK: rock.threadwise_read_into
  // CHECK:  rock.threadwise_accel_gemm
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA * %bufferB from %matrixB features = wmma{
    arch = "amdgcn-amd-amdhsa:gfx1100",
    blockSize = 128 : i32,
    inMPerThread = 2 : i32,
    inNPerThread = 2 : i32,
    params = #rock.wmma_gemm_params<
      mPerBlock = 32,
      nPerBlock = 32,
      kpackPerBlock = 4,
      mPerWave = 16,
      nPerWave = 16,
      kpack = 8,
      splitKFactor = 1,
      forceUnroll = true>
  } : memref<1xvector<8xf32>, #priv> += memref<1xvector<16xf16>, #priv> from memref<32xvector<8xf16>, #wg> * memref<1xvector<16xf16>, #priv> from memref<32xvector<8xf16>, #wg>
  return
}

func.func @rock_blockwise_gemm_accel_wmma_int8(%matrixA : memref<32xvector<16xi8>, #wg>, %matrixB : memref<32xvector<16xi8>, #wg>,
                                               %bufferA : memref<4xvector<16xi8>, #priv>, %bufferB : memref<4xvector<16xi8>, #priv>,
                                               %matrixC : memref<4xvector<8xi32>, #priv>) {
  // CHECK: affine.for {{.*}} = 0 to 2
  // CHECK: rock.threadwise_read_into
  // CHECK: affine.for {{.*}} = 0 to 2
  // CHECK: rock.threadwise_read_into
  // CHECK:  rock.threadwise_accel_gemm
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA * %bufferB from %matrixB features = wmma{
    arch = "amdgcn-amd-amdhsa:gfx1100",
    blockSize = 128 : i32,
    inMPerThread = 2 : i32,
    inNPerThread = 2 : i32,
    params = #rock.wmma_gemm_params<
      mPerBlock = 64,
      nPerBlock = 64,
      kpackPerBlock = 4,
      mPerWave = 32,
      nPerWave = 32,
      kpack = 16,
      splitKFactor = 1,
      forceUnroll = true>
  } : memref<4xvector<8xi32>, #priv> += memref<4xvector<16xi8>, #priv> from memref<32xvector<16xi8>, #wg> * memref<4xvector<16xi8>, #priv> from memref<32xvector<16xi8>, #wg>
  return
}
