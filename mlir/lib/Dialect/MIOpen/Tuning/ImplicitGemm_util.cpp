//===- GridwiseGemmParams.h - MLIR tuning parameter generation --------*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR tuning parameter generation
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpen/Tuning/ImplicitGemm_util.h"

LogicalResult ImplicitGemmUtil::IsValidBlockwiseGemmXdlops(
    const ConvolutionContext &ctx, const int64_t GemmMPerBlock,
    const int64_t GemmNPerBlock, const int64_t GemmKPerBlock,
    const int64_t GemmMPerWave, const int64_t GemmNPerWave,
    const int64_t GemmKPack) {
  // check M, N and K
  std::vector<std::tuple<int64_t, int64_t, int64_t>> validWaveGemmSize = {
      // std::make_tuple(128, 128, 1),
      std::make_tuple(128, 64, 1),
      // std::make_tuple(128, 32, 1),
      // std::make_tuple(128, 16, 1),
      std::make_tuple(64, 128, 1), std::make_tuple(64, 64, 1),
      std::make_tuple(64, 32, 1), std::make_tuple(64, 16, 1),
      // std::make_tuple(32, 128, 1),
      std::make_tuple(32, 64, 1), std::make_tuple(32, 32, 2),
      // std::make_tuple(16, 128, 1),
      std::make_tuple(16, 64, 1), std::make_tuple(16, 16, 4),
      // std::make_tuple(8, 128, 1),
      std::make_tuple(8, 64, 1),
      // std::make_tuple(4, 128, 1),
      std::make_tuple(4, 64, 1)};

  if (!std::any_of(validWaveGemmSize.cbegin(), validWaveGemmSize.cend(),
                   [GemmMPerWave, GemmNPerWave,
                    GemmKPerBlock](const auto it) noexcept -> bool {
                     int64_t validMPerWave, validNPerWave, validKPerWave;
                     std::tie(validMPerWave, validNPerWave, validKPerWave) = it;
                     return (GemmMPerWave == validMPerWave) &&
                            (GemmNPerWave == validNPerWave) &&
                            (GemmKPerBlock % validKPerWave == 0);
                   }))
    return failure();

  const auto WaveSize = 64;
  const auto BlockSize = (GemmNPerBlock * GemmMPerBlock) /
                         (GemmMPerWave * GemmNPerWave) * WaveSize;

  if (BlockSize < 64 || BlockSize > 256)
    return failure();

  if ((GemmMPerBlock % GemmMPerWave) == 0 &&
      (GemmNPerBlock % GemmNPerWave) == 0)
    return success();

  return failure();
}

LogicalResult ImplicitGemmUtil::IsValidGridGemmXdlops(const std::size_t GemmM,
                                                      const std::size_t GemmN,
                                                      const std::size_t GemmK) {
  // unsupported xdlops-gemm
  if (GemmM % 16 != 0 && GemmN % 64 != 0)
    return failure();

  const auto WaveSize = 64;

  if ((GemmM * GemmN) % 256 == 0 && (GemmK * GemmM) % WaveSize == 0 &&
      (GemmK * GemmN) % WaveSize == 0 && GemmN % 16 == 0 && GemmM % 4 == 0 &&
      GemmK % 4 == 0)
    return success();

  return failure();
}

void ImplicitGemmUtil::obtainGemmADimKVectorizable(
    mlir::miopen::ConvOpType opType,
    llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal,
    bool &input1GemmKVectorizable) {
  // Vectorizable flag is opposite between forwad and bwd_data
  if (opType == mlir::miopen::ConvOpType::Conv2DOpType) {
    // When K is not the fastest changing dimension,
    // gemmK dimension is vectorizable, gemmM is not, and vice versa.
    // Vectorization width depending on which among C, Y, X be the fastest
    // changing dimension.
    if (dimIndexVal["k"].first == 3) {
      input1GemmKVectorizable = false;
    } else {
      input1GemmKVectorizable = true;
    }
  } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
    // When K is the fastest changing dimension(3),
    // gemmK dimension is vectorizable, gemmM is not, and vice versa.
    // Vectorization width depending on length of K.
    if (dimIndexVal["k"].first == 3) {
      input1GemmKVectorizable = true;
    } else {
      input1GemmKVectorizable = false;
    }
  } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
    // When K is the fastest changing dimension,
    // gemmM dimension is vectorizable, gemmK is not, and vice versa.
    // Vectorization width depending on which among N, and HoWo be the fastest
    // changing dimension.
    if (dimIndexVal["k"].first == 3) {
      input1GemmKVectorizable = false;
    } else {
      input1GemmKVectorizable = true;
    }
  }
}

void ImplicitGemmUtil::obtainGemmBDimKVectorizable(
    mlir::miopen::ConvOpType opType,
    llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal,
    bool &input2GemmKVectorizable) {
  // Vectorizable flag is opposite between forwad and bwd_data
  if (opType == mlir::miopen::ConvOpType::Conv2DOpType) {
    // For input tensor.
    // When C is the fastest changing dimension,
    // gemmK dimension is vectorizable, gemmN is not, and vice versa.
    // Vectorization width depending on length of C.
    if (dimIndexVal["ci"].first == 3) {
      input2GemmKVectorizable = true;
    } else {
      input2GemmKVectorizable = false;
    }
  } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
    // For output tensor.
    // When K is the fastest changing dimension(3),
    // gemmK dimension is vectorizable, gemmN is not, and vice versa.
    // Vectorization width depending on length of K.
    if (dimIndexVal["ko"].first == 3) {
      input2GemmKVectorizable = true;
    } else {
      input2GemmKVectorizable = false;
    }
  } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
    // For input tensor
    // When C is the fastest changing dimension,
    // gemmN dimension is vectorizable, gemmK is not, and vice versa.
    // Vectorization width depending on length of C.
    if (dimIndexVal["ci"].first == 3) {
      input2GemmKVectorizable = false;
    } else {
      input2GemmKVectorizable = true;
    }
  }
}
