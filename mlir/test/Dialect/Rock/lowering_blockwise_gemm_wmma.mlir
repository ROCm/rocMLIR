// RUN: rocmlir-opt -rock-blockwise-gemm-to-threadwise %s | FileCheck %s
#wg = #gpu.address_space<workgroup>
#priv = #gpu.address_space<private>

func.func @rock_blockwise_gemm_accel_wmma(%matrixA : memref<16xvector<16xf16>, #wg>, %matrixB : memref<16xvector<16xf16>, #wg>,
                                          %bufferA : memref<1xvector<16xf16>, #priv>, %bufferB : memref<1xvector<16xf16>, #priv>,
                                          %matrixC : memref<1xvector<8xf32>, #priv>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[tid:.*]] = rock.workitem_id : index
  // CHECK: %[[wid:.*]] = arith.remui %[[tid]], %c32{{.*}} : index
  // CHECK: {{.*}} = arith.remui %[[wid]], %c16{{.*}} : index
  // CHECK: affine.for {{.*}} = 0 to 1
  // CHECK: affine.for {{.*}} = 0 to 4
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<16xvector<16xf16>, #gpu.address_space<workgroup>>
  // CHECK: memref.store
  // CHECK: affine.for {{.*}} = 0 to 1
  // CHECK: affine.for {{.*}} = 0 to 4
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<16xvector<16xf16>, #gpu.address_space<workgroup>>
  // CHECK: memref.store
  // CHECK:  rock.accel_gemm
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA[%c0] * %bufferB from %matrixB[%c0] features = wmma{
    arch = "amdgcn-amd-amdhsa:gfx1100",
    blockSize = 32 : i32,
    isKContiguousDimA = true,
    isKContiguousDimB = false,
    copyMPerThread = 2 : i32,
    copyNPerThread = 2 : i32,
    params = #rock.wmma_gemm_params<
      kpackPerBlock = 4,
      kpack = 16,
      mPerBlock = 16,
      mPerWave = 16,
      nPerBlock = 16,
      nPerWave = 16,
      forceUnroll = true>
  } : memref<1xvector<8xf32>, #priv> += memref<1xvector<16xf16>, #priv> from memref<16xvector<16xf16>, #wg> * memref<1xvector<16xf16>, #priv> from memref<16xvector<16xf16>, #wg>
  return
}

func.func @rock_blockwise_gemm_accel_wmma_largekpack(%matrixA : memref<32xvector<32xf16>, #wg>, %matrixB : memref<32xvector<32xf16>, #wg>,
                                                     %bufferA : memref<1xvector<16xf16>, #priv>, %bufferB : memref<1xvector<16xf16>, #priv>,
                                                     %matrixC : memref<1xvector<8xf32>, #priv>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[tid:.*]] = rock.workitem_id : index
  // CHECK: %[[wid:.*]] = arith.remui %[[tid]], %c32{{.*}} : index
  // CHECK: {{.*}} = arith.remui %[[wid]], %c16{{.*}} : index
  // CHECK: affine.for {{.*}} = 0 to 1
  // CHECK: affine.for {{.*}} = 0 to 4
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<32xvector<32xf16>, #gpu.address_space<workgroup>>
  // CHECK: {{.*}} = rock.extract_slice %[[a]][{{.*}}] : vector<32xf16> -> vector<16xf16>
  // CHECK: memref.store
  // CHECK: {{.*}} = rock.extract_slice %[[a]][{{.*}}] : vector<32xf16> -> vector<16xf16>
  // CHECK: memref.store
  // CHECK: affine.for {{.*}} = 0 to 1
  // CHECK: affine.for {{.*}} = 0 to 4
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<32xvector<32xf16>, #gpu.address_space<workgroup>>
  // CHECK: {{.*}} = rock.extract_slice %[[b]][{{.*}}] : vector<32xf16> -> vector<16xf16>
  // CHECK: memref.store
  // CHECK: {{.*}} = rock.extract_slice %[[b]][{{.*}}] : vector<32xf16> -> vector<16xf16>
  // CHECK: memref.store
  // CHECK:  rock.accel_gemm
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA[%c0] * %bufferB from %matrixB[%c0] features = wmma{
    arch = "amdgcn-amd-amdhsa:gfx1100",
    blockSize = 128 : i32,
    isKContiguousDimA = true,
    isKContiguousDimB = false,
    copyMPerThread = 2 : i32,
    copyNPerThread = 2 : i32,
    params = #rock.wmma_gemm_params<
      mPerBlock = 32,
      nPerBlock = 32,
      kpackPerBlock = 4,
      mPerWave = 16,
      nPerWave = 16,
      kpack = 32,
      forceUnroll = true>
  } : memref<1xvector<8xf32>, #priv> += memref<1xvector<16xf16>, #priv> from memref<32xvector<32xf16>, #wg> * memref<1xvector<16xf16>, #priv> from memref<32xvector<32xf16>, #wg>
  return
}

func.func @rock_blockwise_gemm_accel_wmma_int8(%matrixA : memref<32xvector<16xi8>, #wg>, %matrixB : memref<32xvector<16xi8>, #wg>,
                                               %bufferA : memref<4xvector<16xi8>, #priv>, %bufferB : memref<4xvector<16xi8>, #priv>,
                                               %matrixC : memref<4xvector<8xi32>, #priv>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[tid:.*]] = rock.workitem_id : index
  // CHECK: %[[wid:.*]] = arith.remui %[[tid]], %c32{{.*}} : index
  // CHECK: {{.*}} = arith.remui %[[wid]], %c16{{.*}} : index
  // CHECK: affine.for {{.*}} = 0 to 2
  // CHECK: affine.for {{.*}} = 0 to 4
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<32xvector<16xi8>, #gpu.address_space<workgroup>>
  // CHECK: memref.store
  // CHECK: affine.for {{.*}} = 0 to 2
  // CHECK: affine.for {{.*}} = 0 to 4
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<32xvector<16xi8>, #gpu.address_space<workgroup>>
  // CHECK: memref.store
  // CHECK:  rock.accel_gemm
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA[%c0] * %bufferB from %matrixB[%c0] features = wmma{
    arch = "amdgcn-amd-amdhsa:gfx1100",
    blockSize = 128 : i32,
    isKContiguousDimA = true,
    isKContiguousDimB = false,
    copyMPerThread = 2 : i32,
    copyNPerThread = 2 : i32,
    params = #rock.wmma_gemm_params<
      mPerBlock = 64,
      nPerBlock = 64,
      kpackPerBlock = 4,
      mPerWave = 32,
      nPerWave = 32,
      kpack = 16,
      forceUnroll = true>
  } : memref<4xvector<8xi32>, #priv> += memref<4xvector<16xi8>, #priv> from memref<32xvector<16xi8>, #wg> * memref<4xvector<16xi8>, #priv> from memref<32xvector<16xi8>, #wg>
  return
}
