// RUN: rocmlir-opt -rock-threadwise-gemm-lowering %s -verify-diagnostics -split-input-file

// Scaled GEMMs require small float types (FP4/FP8/BF8), not F16.
func.func @rock_accel_gemm_wmma_gfx1100_scaled_f16_should_fail(
    %matrixA : memref<1x4xvector<16xf16>, 5>,
    %matrixB : memref<1x4xvector<16xf16>, 5>,
    %matrixC : memref<1x1xvector<8xf32>, 5>,
    %scaleA : memref<1x4xvector<4xf8E8M0FNU>, 5>,
    %scaleB : memref<1x4xvector<4xf8E8M0FNU>, 5>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{For scaled GEMMs, matrixA must be of type Float4E2M1FN, Float8E4M3FN, or Float8E5M2}}
  rock.threadwise_gemm_accel %matrixC += %matrixA scaled by %scaleA * %matrixB scaled by %scaleB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.wmma_gemm_params<
       mPerBlock = 16,
       nPerBlock = 16,
       kpackPerBlock = 4,
       mPerWave = 16,
       nPerWave = 16,
       mnPerXdl = 16,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0,
       forceUnroll = true>
     } : memref<1x1xvector<8xf32>, 5> += memref<1x4xvector<16xf16>, 5> scaled by memref<1x4xvector<4xf8E8M0FNU>, 5> * memref<1x4xvector<16xf16>, 5> scaled by memref<1x4xvector<4xf8E8M0FNU>, 5>
  return
}

// -----

// Scaled GEMMs require small float types (FP4/FP8/BF8), not F16.
func.func @rock_accel_gemm_wmma_gfx1200_scaled_f16_should_fail(
    %matrixA : memref<1x4xvector<8xf16>, 5>,
    %matrixB : memref<1x4xvector<8xf16>, 5>,
    %matrixC : memref<1x1xvector<8xf32>, 5>,
    %scaleA : memref<1x4xvector<4xf8E8M0FNU>, 5>,
    %scaleB : memref<1x4xvector<4xf8E8M0FNU>, 5>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{For scaled GEMMs, matrixA must be of type Float4E2M1FN, Float8E4M3FN, or Float8E5M2}}
  rock.threadwise_gemm_accel %matrixC += %matrixA scaled by %scaleA * %matrixB scaled by %scaleB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1200",
    params = #rock.wmma_gemm_params<
       mPerBlock = 16,
       nPerBlock = 16,
       kpackPerBlock = 4,
       mPerWave = 16,
       nPerWave = 16,
       mnPerXdl = 16,
       kpack = 8,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0,
       forceUnroll = true>
     } : memref<1x1xvector<8xf32>, 5> += memref<1x4xvector<8xf16>, 5> scaled by memref<1x4xvector<4xf8E8M0FNU>, 5> * memref<1x4xvector<8xf16>, 5> scaled by memref<1x4xvector<4xf8E8M0FNU>, 5>
  return
}

// -----

// Scaled WMMA with FP4 is only supported on gfx1250.
func.func @rock_accel_gemm_wmma_gfx1100_scaled_fp4_should_fail(
    %matrixA : memref<4x8xvector<64xf4E2M1FN>, 5>,
    %matrixB : memref<4x8xvector<64xf4E2M1FN>, 5>,
    %matrixC : memref<4x4xvector<8xf32>, 5>,
    %scaleA : memref<4x8xvector<4xf8E8M0FNU>, 5>,
    %scaleB : memref<4x8xvector<4xf8E8M0FNU>, 5>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Wmma supports only F16/BF16/int8 data types}}
  rock.threadwise_gemm_accel %matrixC += %matrixA scaled by %scaleA * %matrixB scaled by %scaleB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.wmma_gemm_params<
       mPerBlock = 64,
       nPerBlock = 64,
       kpackPerBlock = 8,
       mPerWave = 64,
       nPerWave = 64,
       mnPerXdl = 16,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0,
       forceUnroll = true>
     } : memref<4x4xvector<8xf32>, 5> += memref<4x8xvector<64xf4E2M1FN>, 5> scaled by memref<4x8xvector<4xf8E8M0FNU>, 5> * memref<4x8xvector<64xf4E2M1FN>, 5> scaled by memref<4x8xvector<4xf8E8M0FNU>, 5>
  return
}

// -----

// Scaled WMMA with FP4 is only supported on gfx1250.
func.func @rock_accel_gemm_wmma_gfx1200_scaled_fp4_should_fail(
    %matrixA : memref<4x8xvector<64xf4E2M1FN>, 5>,
    %matrixB : memref<4x8xvector<64xf4E2M1FN>, 5>,
    %matrixC : memref<4x4xvector<8xf32>, 5>,
    %scaleA : memref<4x8xvector<4xf8E8M0FNU>, 5>,
    %scaleB : memref<4x8xvector<4xf8E8M0FNU>, 5>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Wmma supports only F16/BF16/int8/E4M3/E5M2 data types}}
  rock.threadwise_gemm_accel %matrixC += %matrixA scaled by %scaleA * %matrixB scaled by %scaleB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1200",
    params = #rock.wmma_gemm_params<
       mPerBlock = 64,
       nPerBlock = 64,
       kpackPerBlock = 8,
       mPerWave = 64,
       nPerWave = 64,
       mnPerXdl = 16,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0,
       forceUnroll = true>
     } : memref<4x4xvector<8xf32>, 5> += memref<4x8xvector<64xf4E2M1FN>, 5> scaled by memref<4x8xvector<4xf8E8M0FNU>, 5> * memref<4x8xvector<64xf4E2M1FN>, 5> scaled by memref<4x8xvector<4xf8E8M0FNU>, 5>
  return
}

// -----

// Scaled WMMA with FP8 is only supported on gfx1250.
func.func @rock_accel_gemm_wmma_gfx1100_scaled_fp8_should_fail(
    %matrixA : memref<4x8xvector<64xf8E4M3FN>, 5>,
    %matrixB : memref<4x8xvector<64xf8E5M2>, 5>,
    %matrixC : memref<4x4xvector<8xf32>, 5>,
    %scaleA : memref<4x8xvector<4xf8E8M0FNU>, 5>,
    %scaleB : memref<4x8xvector<4xf8E8M0FNU>, 5>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Wmma supports only F16/BF16/int8 data types}}
  rock.threadwise_gemm_accel %matrixC += %matrixA scaled by %scaleA * %matrixB scaled by %scaleB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.wmma_gemm_params<
       mPerBlock = 64,
       nPerBlock = 64,
       kpackPerBlock = 8,
       mPerWave = 64,
       nPerWave = 64,
       mnPerXdl = 16,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0,
       forceUnroll = true>
     } : memref<4x4xvector<8xf32>, 5> += memref<4x8xvector<64xf8E4M3FN>, 5> scaled by memref<4x8xvector<4xf8E8M0FNU>, 5> * memref<4x8xvector<64xf8E5M2>, 5> scaled by memref<4x8xvector<4xf8E8M0FNU>, 5>
  return
}

// -----

// Scaled WMMA with mixed FP8/BF8 is only supported on gfx1250.
func.func @rock_accel_gemm_wmma_gfx1200_scaled_fp8_should_fail(
    %matrixA : memref<4x8xvector<64xf8E4M3FN>, 5>,
    %matrixB : memref<4x8xvector<64xf8E5M2>, 5>,
    %matrixC : memref<4x4xvector<8xf32>, 5>,
    %scaleA : memref<4x8xvector<4xf8E8M0FNU>, 5>,
    %scaleB : memref<4x8xvector<4xf8E8M0FNU>, 5>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Wmma does not support mixed types}}
  rock.threadwise_gemm_accel %matrixC += %matrixA scaled by %scaleA * %matrixB scaled by %scaleB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1200",
    params = #rock.wmma_gemm_params<
       mPerBlock = 64,
       nPerBlock = 64,
       kpackPerBlock = 8,
       mPerWave = 64,
       nPerWave = 64,
       mnPerXdl = 16,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0,
       forceUnroll = true>
     } : memref<4x4xvector<8xf32>, 5> += memref<4x8xvector<64xf8E4M3FN>, 5> scaled by memref<4x8xvector<4xf8E8M0FNU>, 5> * memref<4x8xvector<64xf8E5M2>, 5> scaled by memref<4x8xvector<4xf8E8M0FNU>, 5>
  return
}
