// RUN: rocmlir-opt -rock-blockwise-gemm-to-threadwise %s | FileCheck %s

#wg = #gpu.address_space<workgroup>
#priv = #gpu.address_space<private>

// CHECK-LABEL: @rock_blockwise_gemm_accel_two_results
func.func @rock_blockwise_gemm_accel_two_results(%matrixA : memref<256xvector<2xf32>, #wg>, %matrixB : memref<256xvector<2xf32>, #wg>,
                                                %bufferA : memref<4xf32, #priv>, %bufferB : memref<4xf32, #priv>,
                                                %matrixC : memref<4xvector<16xf32>, #priv>) {
  // CHECK:  rock.threadwise_gemm_accel
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA * %bufferB from %matrixB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    blockSize= 256 : i32,
    matrixParamsA = #rock.blockwise_matrix_params<elementType = f32, elementTypeLoad = f32, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
    matrixParamsB = #rock.blockwise_matrix_params<elementType = f32, elementTypeLoad = f32, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
    params = #rock.mfma_gemm_params<
      kpackPerBlock = 2,
      kpack = 2,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<4xvector<16xf32>, #priv> += memref<4xf32, #priv> from memref<256xvector<2xf32>, #wg> * memref<4xf32, #priv> from memref<256xvector<2xf32>, #wg>
  return
}

// CHECK-LABEL: @rock_blockwise_gemm_accel_one_result
func.func @rock_blockwise_gemm_accel_one_result(%matrixA : memref<128xvector<8xi8>, #wg>, %matrixB : memref<128xvector<8xi8>, #wg>,
                                               %bufferA : memref<1xvector<4xi8>, #priv>, %bufferB : memref<1xvector<4xi8>, #priv>,
                                               %matrixC : memref<1xvector<16xi32>, #priv>) {
  // CHECK:  rock.threadwise_gemm_accel
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA * %bufferB from %matrixB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    blockSize = 256 : i32,
    matrixParamsA = #rock.blockwise_matrix_params<elementType = i8, elementTypeLoad = i8, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
    matrixParamsB = #rock.blockwise_matrix_params<elementType = i8, elementTypeLoad = i8, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
    params = #rock.mfma_gemm_params<
      kpackPerBlock = 2,
      kpack = 8,
      mPerBlock = 64,
      mPerWave = 32,
      nPerBlock = 64,
      nPerWave = 32,
      mnPerXdl = 32,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<1xvector<16xi32>, #priv> += memref<1xvector<4xi8>, #priv> from memref<128xvector<8xi8>, #wg> * memref<1xvector<4xi8>, #priv> from memref<128xvector<8xi8>, #wg>
  return
}

// CHECK-LABEL: @rock_blockwise_gemm_accel_fp8_bf8
func.func @rock_blockwise_gemm_accel_fp8_bf8(%matrixA : memref<1024xvector<8xf8E4M3FNUZ>, #gpu.address_space<workgroup>>,
                                          %matrixB : memref<1024xvector<8xf8E5M2FNUZ>, #gpu.address_space<workgroup>>,
                                          %bufferA : memref<4xvector<8xf8E4M3FNUZ>, #gpu.address_space<private>>,
                                          %bufferB : memref<4xvector<8xf8E5M2FNUZ>, #gpu.address_space<private>>,
                                          %matrixC : memref<4xvector<16xf32>, #gpu.address_space<private>>) {
  // CHECK:  rock.threadwise_gemm_accel
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA * %bufferB from %matrixB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx942",
    blockSize = 256 : i32,
    matrixParamsA = #rock.blockwise_matrix_params<elementType = f8E4M3FNUZ, elementTypeLoad = f8E4M3FNUZ, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
    matrixParamsB = #rock.blockwise_matrix_params<elementType = f8E5M2FNUZ, elementTypeLoad = f8E5M2FNUZ, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
    params = #rock.mfma_gemm_params<
      kpackPerBlock = 8,
      mPerBlock = 128,
      nPerBlock = 128,
      kpack = 8,
      mPerWave = 64,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<4xvector<16xf32>, #gpu.address_space<private>> += memref<4xvector<8xf8E4M3FNUZ>, #gpu.address_space<private>> from memref<1024xvector<8xf8E4M3FNUZ>, #gpu.address_space<workgroup>> * memref<4xvector<8xf8E5M2FNUZ>, #gpu.address_space<private>> from memref<1024xvector<8xf8E5M2FNUZ>, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: @rock_blockwise_gemm_accel_fp8_bf8_ocp
func.func @rock_blockwise_gemm_accel_fp8_bf8_ocp(%matrixA : memref<1024xvector<8xf8E4M3FN>, #gpu.address_space<workgroup>>,
                                          %matrixB : memref<1024xvector<8xf8E5M2>, #gpu.address_space<workgroup>>,
                                          %bufferA : memref<4xvector<8xf8E4M3FN>, #gpu.address_space<private>>,
                                          %bufferB : memref<4xvector<8xf8E5M2>, #gpu.address_space<private>>,
                                          %matrixC : memref<4xvector<16xf32>, #gpu.address_space<private>>) {
  // CHECK:  rock.threadwise_gemm_accel
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA * %bufferB from %matrixB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx950",
    blockSize = 256 : i32,
    matrixParamsA = #rock.blockwise_matrix_params<elementType = f8E4M3FN, elementTypeLoad = f8E4M3FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
    matrixParamsB = #rock.blockwise_matrix_params<elementType = f8E5M2, elementTypeLoad = f8E5M2, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
    params = #rock.mfma_gemm_params<
      kpackPerBlock = 8,
      mPerBlock = 128,
      nPerBlock = 128,
      kpack = 8,
      mPerWave = 64,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<4xvector<16xf32>, #gpu.address_space<private>> += memref<4xvector<8xf8E4M3FN>, #gpu.address_space<private>> from memref<1024xvector<8xf8E4M3FN>, #gpu.address_space<workgroup>> * memref<4xvector<8xf8E5M2>, #gpu.address_space<private>> from memref<1024xvector<8xf8E5M2>, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: @rock_blockwise_gemm_accel_fp8_bf8_ocp_double_buffer
func.func @rock_blockwise_gemm_accel_fp8_bf8_ocp_double_buffer(%bufferA : memref<4xvector<8xf8E4M3FN>, #gpu.address_space<private>>,
                                          %bufferB : memref<4xvector<8xf8E5M2>, #gpu.address_space<private>>,
                                          %matrixC : memref<4xvector<16xf32>, #gpu.address_space<private>>) {
  // CHECK: affine.for {{.*}} = 0 to 2
  // CHECK-NOT: rock.threadwise_read_into
  // CHECK: affine.for {{.*}} = 0 to 2
  // CHECK-NOT: rock.threadwise_read_into
  // CHECK: rock.threadwise_gemm_accel
  rock.blockwise_gemm_accel %matrixC += %bufferA * %bufferB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx950",
    blockSize = 256 : i32,
    matrixParamsA = #rock.blockwise_matrix_params<elementType = f8E4M3FN, elementTypeLoad = f8E4M3FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
    matrixParamsB = #rock.blockwise_matrix_params<elementType = f8E5M2, elementTypeLoad = f8E5M2, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
    params = #rock.mfma_gemm_params<
      kpackPerBlock = 8,
      mPerBlock = 128,
      nPerBlock = 128,
      kpack = 8,
      mPerWave = 64,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<4xvector<16xf32>, #gpu.address_space<private>> += memref<4xvector<8xf8E4M3FN>, #gpu.address_space<private>> * memref<4xvector<8xf8E5M2>, #gpu.address_space<private>>
  return
}

// CHECK-LABEL: @rock_blockwise_gemm_accel_scaled_schedule_v2
func.func @rock_blockwise_gemm_accel_scaled_schedule_v2(
    %bufferA : memref<8xvector<32xf4E2M1FN>, #priv>,
    %bufferB : memref<8xvector<32xf4E2M1FN>, #priv>,
    %bufferScaleA : memref<8xvector<32xf8E8M0FNU>, #priv>,
    %bufferScaleB : memref<8xvector<32xf8E8M0FNU>, #priv>,
    %matrixScaleA : memref<512xvector<32xf8E8M0FNU>, #wg>,
    %matrixScaleB : memref<512xvector<32xf8E8M0FNU>, #wg>,
    %matrixC : memref<1xvector<16xf32>, #priv>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK: affine.for
  // CHECK-NOT: rock.threadwise_read_into
  // CHECK: rock.transform {{.*}} : memref<8xvector<32xf4E2M1FN>, #gpu.address_space<private>> to memref<1x8xvector<32xf4E2M1FN>, #gpu.address_space<private>>
  // CHECK: rock.transform {{.*}} : memref<8xvector<32xf8E8M0FNU>, #gpu.address_space<private>> to memref<1x8xvector<32xf8E8M0FNU>, #gpu.address_space<private>>
  // CHECK: affine.for
  // CHECK-NOT: rock.threadwise_read_into
  // CHECK: affine.for
  // CHECK: rock.threadwise_gemm_accel {{.*}} scaled by {{.*}} * {{.*}} scaled by {{.*}} {{.*}} scheduleVersion = 2
  // CHECK: memref<1x1xvector<16xf32>, #gpu.address_space<private>> += memref<1x8xvector<32xf4E2M1FN>, #gpu.address_space<private>> scaled by memref<1x8xvector<32xf8E8M0FNU>, #gpu.address_space<private>> * memref<1x8xvector<32xf4E2M1FN>, #gpu.address_space<private>> scaled by memref<1x8xvector<32xf8E8M0FNU>, #gpu.address_space<private>>
  rock.blockwise_gemm_accel %matrixC += %bufferA scaled by %bufferScaleA from %matrixScaleA
                                      * %bufferB scaled by %bufferScaleB from %matrixScaleB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx950",
    blockSize = 256 : i32,
    matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = true, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 32, inDPerThread = 2>, 
    matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = true, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 32, inDPerThread = 2>,
    params = #rock.xdlops_gemm_derived_params<
      kpackPerBlock = 16,
      kpack = 32,
      mPerBlock = 32,
      mPerWave = 32,
      nPerBlock = 32,
      nPerWave = 32,
      mnPerXdl = 32,
      splitKFactor = 1, 
      scheduleVersion = 2, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<1xvector<16xf32>, #priv> += memref<8xvector<32xf4E2M1FN>, #priv> scaled by memref<8xvector<32xf8E8M0FNU>, #priv> from memref<512xvector<32xf8E8M0FNU>, #wg> * memref<8xvector<32xf4E2M1FN>, #priv> scaled by memref<8xvector<32xf8E8M0FNU>, #priv> from memref<512xvector<32xf8E8M0FNU>, #wg>
  return
}

// CHECK-LABEL: @rock_blockwise_gemm_accel_direct_to_lds
func.func @rock_blockwise_gemm_accel_direct_to_lds(%matrixA : memref<256xvector<2xf32>, #wg>, %matrixB : memref<256xvector<2xf32>, #wg>,
                                                %bufferA : memref<16xi8, #priv>, %bufferB : memref<16xi8, #priv>,
                                                %matrixC : memref<4xvector<16xf32>, #priv>) {

  %c0 = arith.constant 0 : index
  // CHECK:  rock.threadwise_gemm_accel
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA * %bufferB from %matrixB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-",
    blockSize= 256 : i32,
    matrixParamsA = #rock.blockwise_matrix_params<elementType = f32, elementTypeLoad = f32, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = true, directToLDS = true, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
    matrixParamsB = #rock.blockwise_matrix_params<elementType = f32, elementTypeLoad = f32, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = true, directToLDS = true, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
    params = #rock.mfma_gemm_params<
      kpackPerBlock = 2,
      kpack = 2,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1, 
      scheduleVersion = 4, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<4xvector<16xf32>, #priv> += memref<16xi8, #priv> from memref<256xvector<2xf32>, #wg> * memref<16xi8, #priv> from memref<256xvector<2xf32>, #wg>
  return
}
