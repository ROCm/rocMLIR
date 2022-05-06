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

#ifndef MLIR_DIALECT_MIOPEN_GRIDWISE_GEMM_PARAMS_H
#define MLIR_DIALECT_MIOPEN_GRIDWISE_GEMM_PARAMS_H

#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Tuning/ConvContext.h"
#include "mlir/Dialect/MIOpen/Tuning/Serializable.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"
#include "mlir/Dialect/MIOpen/utility/math.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"

#include <cstdint>
#include <string>
#include <unordered_map>

using namespace mlir;

LLVM_YAML_IS_STRING_MAP(int)

// 0 : gemmG dimension.
// 1 : gemmK dimension.
// 2 : gemmM or gemmN dimension.
enum GemmDimensions { GemmG = 0, GemmK = 1, GemmMorN = 2 };

constexpr int64_t gemmCDimG = 0;
constexpr int64_t gemmCDimM = 1;
constexpr int64_t gemmCDimN = 2;
constexpr int64_t gemmCSplitDimM2 = 3;
constexpr int64_t gemmCSplitDimN = 4;
constexpr int64_t gemmCSplitDimN2 = 5;

// greatest common divisor, aka highest common factor
template <typename T> T gcd(T x, T y) {
  if (x == y || x == 0) {
    return y;
  } else if (y == 0) {
    return x;
  } else if (x > y) {
    return gcd(x - y, y);
  } else {
    return gcd(x, y - x);
  }
}

template <typename T, typename... Ys> T gcd(T x, Ys... ys) {
  return gcd(x, gcd(ys...));
}

// least common multiple
template <typename T> T lcm(T x, T y) {
  if (x == 0 || y == 0) {
    return 0;
  } else {
    return (x * y) / gcd(x, y);
  }
}

template <typename T, typename... Ys> T lcm(T x, Ys... ys) {
  return lcm(x, lcm(ys...));
}

template <typename T> T integer_divide_ceil(T x, T y) {
  return (x + y - 1) / y;
}

template <typename T> T integer_least_multiple(T x, T y) {
  return y * integer_divide_ceil(x, y);
}

struct InitParams {
  int64_t gemmMPerBlock;
  int64_t gemmNPerBlock;
  int64_t gemmKPerBlock;
};

struct GemmSize {
  int64_t gemmG;
  int64_t gemmM;
  int64_t gemmN;
  int64_t gemmK;
};

struct DerivedParams {
  int64_t srcVectorReadDim;
  int64_t srcDataPerRead;
  int64_t dstDataPerWrite;
  int64_t clusterLenGemmPos0; // G
  int64_t clusterLenGemmPos1; // K
  int64_t clusterLenGemmPos2; // M or N
  DerivedParams()
      : srcVectorReadDim(GemmG), srcDataPerRead(1), dstDataPerWrite(1),
        clusterLenGemmPos1(0), clusterLenGemmPos2(0) {}
};

struct DerivedOutParams {
  int64_t gemmVectorDim;
  int64_t destVectorDim;
  int64_t dataPerCopy;
  DerivedOutParams() : gemmVectorDim(-1), destVectorDim(-1), dataPerCopy(1) {}
};

class PopulateParamsBase {
public:
  static void
  obtainGemmADimKVectorizable(mlir::miopen::ConvOpType opType,
                              llvm::StringMap<DimIndexAndSize> &dimIndexAndSize,
                              bool &input1GemmKVectorizable) {
    // Vectorizable flag is opposite between forwad and bwd_data
    if (opType == mlir::miopen::ConvOpType::Fwd) {
      // When K is not the fastest changing dimension,
      // gemmK dimension is vectorizable, gemmM is not, and vice versa.
      // Vectorization width depending on which among C, Y, X be the fastest
      // changing dimension.
      if (dimIndexAndSize["k"].index == 4) {
        input1GemmKVectorizable = false;
      } else {
        input1GemmKVectorizable = true;
      }
    } else if (opType == mlir::miopen::ConvOpType::BwdData) {
      // always load gemmM first
      input1GemmKVectorizable = false;
    } else if (opType == mlir::miopen::ConvOpType::BwdWeight) {
      // When K is the fastest changing dimension,
      // gemmM dimension is vectorizable, gemmK is not, and vice versa.
      // Vectorization width depending on which among N, and HoWo be the fastest
      // changing dimension.
      if (dimIndexAndSize["k"].index == 4) {
        input1GemmKVectorizable = false;
      } else {
        input1GemmKVectorizable = true;
      }
    }
  }

  static void
  obtainGemmBDimKVectorizable(mlir::miopen::ConvOpType opType,
                              llvm::StringMap<DimIndexAndSize> &dimIndexAndSize,
                              bool &input2GemmKVectorizable) {
    // Vectorizable flag is opposite between forwad and bwd_data
    if (opType == mlir::miopen::ConvOpType::Fwd) {
      // For input tensor.
      // When C is the fastest changing dimension,
      // gemmK dimension is vectorizable, gemmN is not, and vice versa.
      // Vectorization width depending on length of C.
      if (dimIndexAndSize["ci"].index == 4) {
        input2GemmKVectorizable = true;
      } else {
        input2GemmKVectorizable = false;
      }
    } else if (opType == mlir::miopen::ConvOpType::BwdData) {
      // For output tensor.
      // When K is the fastest changing dimension(3),
      // gemmK dimension is vectorizable, gemmN is not, and vice versa.
      // Vectorization width depending on length of K.
      if (dimIndexAndSize["ko"].index == 4) {
        input2GemmKVectorizable = true;
      } else {
        input2GemmKVectorizable = false;
      }
    } else if (opType == mlir::miopen::ConvOpType::BwdWeight) {
      // For input tensor
      // When C is the fastest changing dimension,
      // gemmN dimension is vectorizable, gemmK is not, and vice versa.
      // Vectorization width depending on length of C.
      if (dimIndexAndSize["ci"].index == 4) {
        input2GemmKVectorizable = false;
      } else {
        input2GemmKVectorizable = true;
      }
    }
  }

  static void obtainFilterVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto dimIndexAndSize = ctx.dimIndexAndSize;
    // Vectorization length logic is the same for forward and bwd_data
    if (dimIndexAndSize["k"].index == 4) {
      vecLen = dimIndexAndSize["k"].size;
    } else if (dimIndexAndSize["k"].index == 1) {
      // dimKF is the lowest changing dimension, which means dimC/dimY/dimX
      vecLen = dimIndexAndSize["c"].size * dimIndexAndSize["y"].size *
               dimIndexAndSize["x"].size;
    } else if (dimIndexAndSize["k"].index == 2) {
      // K's position is at 2, vectorization legnth is last two dimension
      if (dimIndexAndSize["c"].index == 1) {
        vecLen = dimIndexAndSize["y"].size * dimIndexAndSize["x"].size;
      } else if (dimIndexAndSize["y"].index == 1) {
        vecLen = dimIndexAndSize["c"].size * dimIndexAndSize["x"].size;
      } else {
        vecLen = dimIndexAndSize["c"].size * dimIndexAndSize["y"].size;
      }
    } else {
      // K's position is 3, vectorization legnth is last dimension
      if (dimIndexAndSize["c"].index == 4) {
        vecLen = dimIndexAndSize["c"].size;
      } else if (dimIndexAndSize["y"].index == 4) {
        vecLen = dimIndexAndSize["y"].size;
      } else {
        vecLen = dimIndexAndSize["x"].size;
      }
    }
  }

  static void obtainBwdDataFilterVecLen(ConvolutionContext &ctx,
                                        int64_t &vecLen) {
    auto dimIndexAndSize = ctx.dimIndexAndSize;
    // Vectorization length logic is the same for forward and bwd_data
    if (dimIndexAndSize["c"].index == 4) {
      vecLen = dimIndexAndSize["c"].size;
    } else if (dimIndexAndSize["c"].index == 2) {
      // C's position is at 2, vectorization legnth depend last two dimension
      if (dimIndexAndSize["y"].size == 1 && dimIndexAndSize["x"].size == 1) {
        vecLen = dimIndexAndSize["c"].size;
      } else {
        vecLen = 1;
      }
    } else {
      vecLen = 1;
    }
  }
  static void obtainInputVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto dimIndexAndSize = ctx.dimIndexAndSize;
    if (dimIndexAndSize["ni"].index == 4) {
      vecLen = dimIndexAndSize["ni"].size;
    } else if (dimIndexAndSize["ci"].index == 4) {
      vecLen = dimIndexAndSize["ci"].size;
    } else {
      if (dimIndexAndSize["x"].size == 1 && dimIndexAndSize["y"].size == 1 &&
          ctx.strideVal[0] == 1 && ctx.strideVal[1] == 1 &&
          ctx.paddingVal[0] == 0 && ctx.paddingVal[1] == 0 &&
          ctx.paddingVal[2] == 0 && ctx.paddingVal[3] == 0)
        vecLen = dimIndexAndSize["ho"].size * dimIndexAndSize["wo"].size;
      else
        vecLen = 1;
    }
  }
  static void obtainBwdDataOutputVecLen(ConvolutionContext &ctx,
                                        int64_t &vecLen) {
    auto dimIndexAndSize = ctx.dimIndexAndSize;
    if (dimIndexAndSize["ko"].index == 4) {
      vecLen = dimIndexAndSize["ko"].size;
    } else if (dimIndexAndSize["no"].index == 4) {
      vecLen = dimIndexAndSize["no"].size;
    } else if (dimIndexAndSize["no"].index == 0) {
      if (dimIndexAndSize["ho"].index == 3 &&
          dimIndexAndSize["wo"].index == 4) {
        if (dimIndexAndSize["y"].size == 1 && dimIndexAndSize["x"].size == 1)
          vecLen = dimIndexAndSize["ho"].size * dimIndexAndSize["wo"].size;
        else
          vecLen = 1;
      } else
        vecLen = 1;
    } else {
      vecLen = 1;
    }
  }

  static void obtainOutputVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto dimIndexAndSize = ctx.dimIndexAndSize;
    if (dimIndexAndSize["ko"].index == 4) {
      vecLen = dimIndexAndSize["ko"].size;
    } else if (dimIndexAndSize["ko"].index == 1) {
      // dimKO is the lowest changing dimension, which means dimN/dimHo/dimWo
      vecLen = dimIndexAndSize["no"].size * dimIndexAndSize["ho"].size *
               dimIndexAndSize["wo"].size;
    } else if (dimIndexAndSize["ko"].index == 2) {
      // Ko's position is at 2, vectorization legnth is last two dimensions
      if (dimIndexAndSize["no"].index == 0) {
        vecLen = dimIndexAndSize["ho"].size * dimIndexAndSize["wo"].size;
      } else if (dimIndexAndSize["ho"].index == 0) {
        vecLen = dimIndexAndSize["no"].size * dimIndexAndSize["wo"].size;
      } else {
        vecLen = dimIndexAndSize["no"].size * dimIndexAndSize["ho"].size;
      }
    } else {
      // K's position is 3, vectorization legnth is last dimension
      if (dimIndexAndSize["no"].index == 4) {
        vecLen = dimIndexAndSize["no"].size;
      } else if (dimIndexAndSize["ho"].index == 4) {
        vecLen = dimIndexAndSize["ho"].size;
      } else {
        vecLen = dimIndexAndSize["wo"].size;
      }
    }
  }

  static void obtainGemmAVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto opType = ctx.opType;
    if (opType == mlir::miopen::ConvOpType::Fwd) {
      obtainFilterVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::BwdData) {
      obtainBwdDataFilterVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::BwdWeight) {
      obtainOutputVecLen(ctx, vecLen);
    }
  }

  static void obtainGemmBVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto opType = ctx.opType;
    if (opType == mlir::miopen::ConvOpType::Fwd) {
      obtainInputVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::BwdData) {
      obtainBwdDataOutputVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::BwdWeight) {
      obtainInputVecLen(ctx, vecLen);
    }
  }

  static void obtainGemmCVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto opType = ctx.opType;
    if (opType == mlir::miopen::ConvOpType::Fwd) {
      obtainOutputVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::BwdData) {
      obtainInputVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::BwdWeight) {
      obtainFilterVecLen(ctx, vecLen);
    }
  }

protected:
  mlir::LogicalResult calculateInputDerivedParams(InitParams *param,
                                                  int64_t blockSize,
                                                  ConvolutionContext &ctx,
                                                  bool isGemmA,
                                                  DerivedParams &derived) {

    bool gemmKVectorizable = false;
    int64_t vectorizableLength = 0;
    if (isGemmA) {
      obtainGemmADimKVectorizable(ctx.opType, ctx.dimIndexAndSize,
                                  gemmKVectorizable);
      obtainGemmAVecLen(ctx, vectorizableLength);
    } else {
      obtainGemmBDimKVectorizable(ctx.opType, ctx.dimIndexAndSize,
                                  gemmKVectorizable);
      obtainGemmBVecLen(ctx, vectorizableLength);
    }

    // calculate threadwise copy size
    int64_t dataPerThreadCopy = 0;
    if (isGemmA) {
      dataPerThreadCopy =
          (param->gemmKPerBlock * param->gemmMPerBlock) / blockSize;
    } else {
      dataPerThreadCopy =
          (param->gemmKPerBlock * param->gemmNPerBlock) / blockSize;
    }

    if (!(dataPerThreadCopy > 0))
      return mlir::failure();

    // Compute the maximum possible vectorization size for the data type used
    // in the algorithm.
    int64_t vectorizationSize = 1;
    auto dataType = ctx.getDataType();

    // TODO: Revert the vectorizationSize decision with below commented code:
    // unsigned dataWidth = dataType.getIntOrFloatBitWidth();
    // const size_t highestPotentialVectorizationLen = 128;
    // vectorizationSize = highestPotentialVectorizationLen / dataWidth;
    if (dataType.isF32()) {
      vectorizationSize = 4;
    } else if (dataType.isF16() || dataType.isBF16()) {
      // Nonxdlops on fp16 resnet50 fail for vectorization size > 4
      // Xdlops is okay on 4, 8
      vectorizationSize = 4;
    } else if (dataType.isInteger(8)) {
      // Nonxdlops on in8 resnet50 fail for vectorization size > 4
      // Xdlops is okay on 4, 8, 16
      vectorizationSize = 4;
    }

    // FIXME: set vectorizationSize be 1 for backward data and backward
    // weight for now.
    // The logic for deciding vectorization size and dimension for
    // backward data and backward weight has to be reviewed.
    auto opType = ctx.opType;
    if (opType == mlir::miopen::ConvOpType::BwdData ||
        opType == mlir::miopen::ConvOpType::BwdWeight) {
      vectorizationSize = 1;
    }

    // srcDataPerRead bounded by size of threadwise copy
    if ((vectorizableLength > 0) && (vectorizableLength % 4 == 0)) {
      derived.srcDataPerRead = gcd(vectorizationSize, dataPerThreadCopy);
    }

    // decide threadwise copy lengths
    const auto dataPerThreadCopyGemmVectorized = derived.srcDataPerRead;
    const auto dataPerThreadCopyGemmNonvectorized =
        dataPerThreadCopy / dataPerThreadCopyGemmVectorized;

    int64_t dataPerThreadCopyGemmPos1 = 0;
    int64_t dataPerThreadCopyGemmPos2 = 0;
    if (gemmKVectorizable) {
      dataPerThreadCopyGemmPos1 = dataPerThreadCopyGemmVectorized;
      dataPerThreadCopyGemmPos2 = dataPerThreadCopyGemmNonvectorized;
      derived.srcVectorReadDim = GemmK;
    } else {
      dataPerThreadCopyGemmPos1 = dataPerThreadCopyGemmNonvectorized;
      dataPerThreadCopyGemmPos2 = dataPerThreadCopyGemmVectorized;
      derived.srcVectorReadDim = GemmMorN;
    }
    assert(derived.srcVectorReadDim != GemmG);

    // FIXME: force scalar write for now. Logic being commented out would
    // need to be scrutinized.
    //
    // dstDataPerWrite also bounded by size of threadwise copy
    // derived.dstDataPerWrite = gcd(vectorizationSize,
    // dataPerThreadCopyGemmPos2);
    derived.dstDataPerWrite = 1;

    // calculate blockwise copy thread cluster lengths
    if (isGemmA) {
      derived.clusterLenGemmPos1 =
          param->gemmKPerBlock / dataPerThreadCopyGemmPos1;
      derived.clusterLenGemmPos2 =
          param->gemmMPerBlock / dataPerThreadCopyGemmPos2;
    } else {
      derived.clusterLenGemmPos1 =
          param->gemmKPerBlock / dataPerThreadCopyGemmPos1;
      derived.clusterLenGemmPos2 =
          param->gemmNPerBlock / dataPerThreadCopyGemmPos2;
    }
    if (!(derived.clusterLenGemmPos1 > 0 && derived.clusterLenGemmPos2 > 0))
      return mlir::failure();

    return mlir::success();
  }

  mlir::LogicalResult calculateOutputDerivedParams(InitParams *params,
                                                   int64_t blockSize,
                                                   ConvolutionContext &ctx,
                                                   DerivedOutParams &out) {
    int64_t cVectorLength = 0;
    mlir::miopen::ConvOpType op = ctx.getOpType();

    obtainGemmCVecLen(ctx, cVectorLength);
    int64_t dataPerThread =
        (params->gemmMPerBlock * params->gemmNPerBlock) / blockSize;
    if (!(dataPerThread > 0)) {
      return failure();
    }

    // TODO: Allow vectorization group size of 2
    int64_t vectorizationSize = 4;
    // No swizzling or vectorization for backward data
    // TODO(kdrewnia): Understand when it might be possible
    if (mlir::miopen::ConvOpType::BwdData == op) {
      vectorizationSize = 1;
    }

    if ((cVectorLength > 0) && (dataPerThread % vectorizationSize == 0) &&
        (cVectorLength % vectorizationSize == 0)) {
      out.dataPerCopy = gcd(dataPerThread, vectorizationSize);
    } else {
      out.dataPerCopy = 1;
    }

    auto &dimIndexAndSize = ctx.dimIndexAndSize;
    // Find dimensions in which the copy will take place
    switch (op) {
    case mlir::miopen::ConvOpType::Fwd:
      if (dimIndexAndSize["ko"].index == 4) {
        out.gemmVectorDim = gemmCDimM;
        out.destVectorDim = 4;
      } else {
        out.gemmVectorDim = gemmCDimN;
        // This relies on assumptions about how we load our data for GEMM
        out.destVectorDim = dimIndexAndSize["wo"].index;
      }
      break;
    case mlir::miopen::ConvOpType::BwdWeight:
      if (dimIndexAndSize["k"].index == 4) {
        out.gemmVectorDim = gemmCDimM;
        out.destVectorDim = 4;
      } else {
        out.gemmVectorDim = gemmCDimN;
        // Backward weight computations fold the {c, y, x} dimensions
        // into N using the native order
        out.destVectorDim = 4;
      }
      break;
    case mlir::miopen::ConvOpType::BwdData:
      out.gemmVectorDim = -1;
      out.destVectorDim = -1;
      break;
    }
    return success();
  }

  static void obtainGemmSize(ConvolutionContext &ctx, GemmSize &gemmSize) {
    gemmSize.gemmG = ctx.dimIndexAndSize["g"].size;

    if (ctx.opType == mlir::miopen::ConvOpType::Fwd) {
      gemmSize.gemmM = ctx.dimIndexAndSize["k"].size;
      gemmSize.gemmN = ctx.dimIndexAndSize["no"].size *
                       ctx.dimIndexAndSize["ho"].size *
                       ctx.dimIndexAndSize["wo"].size;
      gemmSize.gemmK = ctx.dimIndexAndSize["c"].size *
                       ctx.dimIndexAndSize["y"].size *
                       ctx.dimIndexAndSize["x"].size;
    } else if (ctx.opType == mlir::miopen::ConvOpType::BwdData) {
      int64_t y, x, ho, wo, hi, wi;
      y = x = ho = wo = hi = wi = 0;
      y = ctx.dimIndexAndSize["y"].size;
      x = ctx.dimIndexAndSize["x"].size;
      ho = ctx.dimIndexAndSize["ho"].size;
      wo = ctx.dimIndexAndSize["wo"].size;
      hi = ctx.dimIndexAndSize["hi"].size;
      wi = ctx.dimIndexAndSize["wi"].size;
      auto strideH = ctx.strideVal[0];
      auto strideW = ctx.strideVal[1];
      auto dilationH = ctx.dilationVal[0];
      auto dilationW = ctx.dilationVal[1];
      auto leftPadH = ctx.paddingVal[0];
      auto leftPadW = ctx.paddingVal[1];

      auto gcdStrideDilationH = math_util::gcd(strideH, dilationH);
      auto gcdStrideDilationW = math_util::gcd(strideW, dilationW);

      auto yTilda = strideH / gcdStrideDilationH;
      auto xTilda = strideW / gcdStrideDilationW;

      auto hTilda =
          ho + math_util::integer_divide_ceil(dilationH * (y - 1), strideH);
      auto wTilda =
          wo + math_util::integer_divide_ceil(dilationW * (x - 1), strideW);

      auto iHTildaLeft = math_util::integer_divide_floor(
          std::max(0l, leftPadH - dilationH * (yTilda - 1)), strideH);
      auto iWTildaLeft = math_util::integer_divide_floor(
          std::max(0l, leftPadW - dilationW * (xTilda - 1)), strideW);

      auto iHTildaRight = std::min(
          hTilda, math_util::integer_divide_ceil(leftPadH + hi - 1, strideH) + 1);
      auto iWTildaRight = std::min(
          wTilda, math_util::integer_divide_ceil(leftPadW + wi - 1, strideW) + 1);

      auto hTildaSlice = iHTildaRight - iHTildaLeft;
      auto wTildaSlice = iWTildaRight - iWTildaLeft;

      auto gemmId = ctx.gemmId;
      auto iYTilda = gemmId / xTilda;
      auto iXTilda = gemmId % xTilda;
      auto yDotSlice = math_util::integer_divide_ceil(y - iYTilda, yTilda);
      auto xDotSlice = math_util::integer_divide_ceil(x - iXTilda, xTilda);

      gemmSize.gemmM = ctx.dimIndexAndSize["c"].size;
      gemmSize.gemmN =
          ctx.dimIndexAndSize["no"].size * hTildaSlice * wTildaSlice;
      gemmSize.gemmK = ctx.dimIndexAndSize["k"].size * yDotSlice * xDotSlice;
    } else if (ctx.opType == mlir::miopen::ConvOpType::BwdWeight) {
      gemmSize.gemmM = ctx.dimIndexAndSize["k"].size;
      gemmSize.gemmK = ctx.dimIndexAndSize["no"].size *
                       ctx.dimIndexAndSize["ho"].size *
                       ctx.dimIndexAndSize["wo"].size;
      gemmSize.gemmN = ctx.dimIndexAndSize["c"].size *
                       ctx.dimIndexAndSize["y"].size *
                       ctx.dimIndexAndSize["x"].size;
    }
  }

  int64_t obtainGridSize(GemmSize &gemmSize, InitParams *param) {
    return (gemmSize.gemmM / param->gemmMPerBlock) *
           (gemmSize.gemmN / param->gemmNPerBlock) * gemmSize.gemmG;
  }
};

struct InitParamsNonXDL : InitParams, Serializable<InitParamsNonXDL> {
  InitParamsNonXDL(int64_t bSize, int64_t mPerBlock, int64_t nPerBlock,
                   int64_t kPerBlock, int64_t mPerThread, int64_t nPerThread)
      : InitParams{mPerBlock, nPerBlock, kPerBlock}, gemmMPerThread(mPerThread),
        gemmNPerThread(nPerThread), blockSize(bSize) {}
  int64_t gemmMPerThread;
  int64_t gemmNPerThread;
  int64_t blockSize;

  InitParamsNonXDL() : InitParamsNonXDL(0LL, 0LL, 0LL, 0LL, 0LL, 0LL) {}

  template <class Self, class F> static void visit(Self &&self, F f) {
    f(self.blockSize);
    f(self.gemmMPerBlock);
    f(self.gemmNPerBlock);
    f(self.gemmKPerBlock);
    f(self.gemmMPerThread);
    f(self.gemmNPerThread);
  }
};

struct InitParamsXDL : InitParams, Serializable<InitParamsXDL> {
  InitParamsXDL(int64_t mPerBlock, int64_t nPerBlock, int64_t kPerBlock,
                int64_t mPerWave, int64_t nPerWave, int64_t kPack,
                bool aThreadCopyMoreGemmK, bool bThreadCopyMoreGemmKPack)
      : InitParams{mPerBlock, nPerBlock, kPerBlock}, gemmMPerWave(mPerWave),
        gemmNPerWave(nPerWave), gemmKPack(kPack),
        gemmAThreadCopyMoreGemmK(aThreadCopyMoreGemmK),
        gemmBThreadCopyMoreGemmKPack(bThreadCopyMoreGemmKPack) {}

  InitParamsXDL() : InitParamsXDL(0LL, 0LL, 0LL, 0LL, 0LL, 0LL, false, false) {}

  int64_t gemmMPerWave;
  int64_t gemmNPerWave;
  int64_t gemmKPack;
  bool gemmAThreadCopyMoreGemmK;
  bool gemmBThreadCopyMoreGemmKPack;

  template <class Self, class F> static void visit(Self &&self, F f) {
    f(self.gemmMPerBlock);
    f(self.gemmNPerBlock);
    f(self.gemmKPerBlock);
    f(self.gemmMPerWave);
    f(self.gemmNPerWave);
    f(self.gemmKPack);
    f(self.gemmAThreadCopyMoreGemmK);
    f(self.gemmBThreadCopyMoreGemmKPack);
  }
};

template <typename T> std::string genDebugForParams(T params) {
  std::ostringstream os;
  params.visit(params, [&os](auto &arg) { os << arg << ","; });
  os << "\n";
  return os.str();
}

// block gemm tuning params that sepcific the layout of thread-wise gemm in a
// workgroup
struct DerivedBlockGemmParams {
  int64_t gemmMLevel0Cluster;
  int64_t gemmNLevel0Cluster;
  int64_t gemmMLevel1Cluster;
  int64_t gemmNLevel1Cluster;
};

class PopulateParams : public PopulateParamsBase {
private:
  // clang-format off
  llvm::SmallVector<InitParamsNonXDL, 8> initParameters = {
    // blockSize M/block N/block K/block M/thread N/thread
    {256, 128, 128, 16, 4, 4},
    {256, 128, 128, 8, 4, 4},
    {256, 128, 128, 4, 4, 4},
    {128, 128, 64, 16, 4, 4},
    {128, 128, 64, 8, 4, 4},
    {128, 128, 64, 4, 4, 4},
    {128, 64, 128, 16, 4, 4},
    {128, 64, 128, 8, 4, 4},
    {128, 64, 128, 4, 4, 4},
    {64, 64, 64, 16, 4, 4},
    {64, 64, 64, 8, 4, 4},
    {64, 64, 64, 4, 4, 4},
    {64, 64, 32, 16, 4, 2},
    {64, 64, 32, 8, 4, 2},
    {64, 64, 32, 4, 4, 2},
    {64, 32, 64, 16, 2, 4},
    {64, 32, 64, 8, 2, 4},
    {64, 32, 64, 4, 2, 4},
    {64, 32, 32, 16, 2, 2},
    {64, 32, 32, 8, 2, 2},
    {64, 32, 32, 4, 2, 2}};
  // clang-format on

  // if can't select config from above , use this config to do
  // padding kernel for example , GemmK/block is 16 , if your gemmK is  13 , we
  // add more 3 gemmk
  InitParams universal_Parameters = {64, 64, 16};

  LogicalResult
  calculateGemmABlockCopyPerformanceParameters(InitParamsNonXDL *param,
                                               ConvolutionContext &ctx,
                                               DerivedParams &derived) {
    return calculateInputDerivedParams(param, param->blockSize, ctx, true,
                                       derived);
  }

  LogicalResult
  calculateGemmBBlockCopyPerformanceParameters(InitParamsNonXDL *param,
                                               ConvolutionContext &ctx,
                                               DerivedParams &derived) {

    return calculateInputDerivedParams(param, param->blockSize, ctx, false,
                                       derived);
  }

  LogicalResult calculateGemmCBlockwiseCopyParams(InitParamsNonXDL *params,
                                                  ConvolutionContext &ctx,
                                                  DerivedOutParams &out) {
    return calculateOutputDerivedParams(params, params->blockSize, ctx, out);
  }

  LogicalResult
  CalculateBlockGemmPerformanceParameters(const InitParamsNonXDL &param,
                                          const ConvolutionContext &ctx,
                                          DerivedBlockGemmParams &derived) {

    derived.gemmMLevel0Cluster = 0;
    derived.gemmNLevel0Cluster = 0;
    derived.gemmMLevel1Cluster = 0;
    derived.gemmNLevel1Cluster = 0;

    if (param.blockSize == 64) {
      derived.gemmMLevel0Cluster = 4;
      derived.gemmNLevel0Cluster = 4;
      derived.gemmMLevel1Cluster = 2;
      derived.gemmNLevel1Cluster = 2;
    } else if (param.blockSize == 128) {
      derived.gemmMLevel0Cluster = 4;
      derived.gemmNLevel0Cluster = 4;
      derived.gemmMLevel1Cluster = 4;
      derived.gemmNLevel1Cluster = 2;
    } else if (param.blockSize == 256) {
      derived.gemmMLevel0Cluster = 4;
      derived.gemmNLevel0Cluster = 4;
      derived.gemmMLevel1Cluster = 4;
      derived.gemmNLevel1Cluster = 4;
    } else {
      return failure();
    }

    if (!(param.gemmMPerThread >= 2 && param.gemmMPerThread <= 4))
      return failure();

    if (!(param.gemmNPerThread >= 2 && param.gemmNPerThread <= 4))
      return failure();

    if (!(param.gemmMPerBlock % param.gemmMPerThread == 0 &&
          param.gemmNPerBlock % param.gemmNPerThread == 0))
      return failure();

    const auto threadGemmMPerBlock = param.gemmMPerBlock / param.gemmMPerThread;
    const auto threadGemmNPerBlock = param.gemmNPerBlock / param.gemmNPerThread;

    const auto threadGemmMPerCluster =
        derived.gemmMLevel0Cluster * derived.gemmMLevel1Cluster;
    const auto threadGemmNPerCluster =
        derived.gemmNLevel0Cluster * derived.gemmNLevel1Cluster;

    if (!(threadGemmMPerBlock % threadGemmMPerCluster == 0) &&
        (threadGemmNPerBlock % threadGemmNPerCluster == 0))
      return failure();

    const auto clusterMPerBlock = threadGemmMPerBlock / threadGemmMPerCluster;
    const auto clusterNPerBlock = threadGemmNPerBlock / threadGemmNPerCluster;

    // inline asm only support clusterMPerBlock = 2 andclusterNPerBlock =
    // 2
    if (!(clusterMPerBlock == 2 && clusterNPerBlock == 2))
      return failure();

    return success();
  }

  LogicalResult
  populateDerived(ConvolutionContext &ctx, InitParamsNonXDL &validParams,
                  GemmSize &gemmSize, DerivedParams &gemmADerivedParam,
                  DerivedParams &gemmBDerivedParam,
                  DerivedBlockGemmParams &blockGemmDerivedParam,
                  DerivedOutParams &gemmCDerivedParam, int64_t &gridSize);

  LogicalResult populatePaddingKernelDerived(
      ConvolutionContext &ctx, InitParamsNonXDL &validParams,
      GemmSize &gemmSize, DerivedParams &gemmADerivedParam,
      DerivedParams &gemmBDerivedParam,
      DerivedBlockGemmParams &blockGemmDerivedParam,
      DerivedOutParams &gemmCDerivedParam, int64_t &gridSize);

public:
  LogicalResult obtainTuningParameters(
      Operation *op, int64_t blockSizeOverride, const std::string &perfConfig,
      InitParamsNonXDL &validParams, DerivedParams &gemmADerivedParam,
      DerivedParams &gemmBDerivedParam,
      DerivedBlockGemmParams &blockGemmDerivedParam,
      DerivedOutParams &gemmCDerivedParam, int64_t &gridSize);

  llvm::SmallVector<InitParamsNonXDL, 8>
  getTuningParameters(miopen::ConvOpType dir, mlir::Type dataType) {
    return initParameters;
  }

  InitParams getUniversalParameters() { return universal_Parameters; }

  LogicalResult isValidGemm(InitParamsNonXDL *param, GemmSize &gemmSize) {
    if (!(gemmSize.gemmM % param->gemmMPerBlock == 0 &&
          gemmSize.gemmN % param->gemmNPerBlock == 0 &&
          gemmSize.gemmK % param->gemmKPerBlock == 0)) {
      return mlir::failure();
    }
    return mlir::success();
  }
};

class PopulateParamsXDL : public PopulateParamsBase {
private:
  // Initial tuning parameters for forward convolution and backward
  // convolution.
  // clang-format off
  llvm::SmallVector<InitParamsXDL, 4> initParameters = {
      // M/block N/block K/block M/wave N/wave kPack aCopyMore bCopyMore
      {128, 128, 4, 64, 64, 4, false, false},
      {32, 64, 4, 32, 64, 4, false, false},

      {128, 128, 8, 64, 64, 1, false, false},
      {128, 128, 16, 64, 64, 1, false, false},
      {8, 64, 8, 8, 64, 1, false, false},
      {4, 64, 16, 4, 64, 1, false, false},
      {32, 64, 4, 32, 64, 1, false, false},
      {16, 16, 16, 16, 16, 1, false, false},
      {16, 16, 4, 16, 16, 1, false, false},
  };
  // clang-format on

  // clang-format off
  llvm::SmallVector<InitParamsXDL, 4> initParametersForwardI8 = {
      // M/block N/block K/block M/wave N/wave kPack aCopyMore bCopyMore
      // kpack for int8 must be larger than kbase, which means
      // kpack must be at least 4, once enabled.
      {64, 64, 8, 32, 32, 8, false, false},
      {64, 64, 8, 32, 32, 4, false, false},
      {32, 32, 8, 16, 16, 8, false, false},
      {32, 32, 8, 16, 16, 4, false, false},
      // The 32 x 32 xdlops k/block must be at least 8
      {64, 64, 16, 32, 32, 1, false, false},
      {64, 64, 8, 32, 32, 1, false, false},
      {32, 32, 16, 32, 32, 1, false, false},
      {32, 32, 8, 32, 32, 1, false, false},
      // The 16 x 16 xdlops k/block must be at least 16
      {32, 32, 32, 16, 16, 1, false, false},
      {32, 32, 16, 16, 16, 1, false, false},
      {16, 16, 32, 16, 16, 1, false, false},
      {16, 16, 16, 16, 16, 1, false, false},
  };
  // clang-format on

  // XXX FIXME: Initial tuning parameters for fp32 backward weight convolution.
  // Deliberately avoid KPACK=4 for fp32 backward weight convolution. It has
  // been verified some configs would cause intermittent failures.
  // TODO(whchung): Get to the bottom of this.
  // clang-format off
  llvm::SmallVector<InitParamsXDL, 4> initParametersBwdWeightF32 = {
      // M/block N/block K/block M/wave N/wave kPack aCopyMore bCopyMore
      {128, 128, 8, 64, 64, 1, false, false},
      {128, 128, 16, 64, 64, 1, false, false},
      {8, 64, 8, 8, 64, 1, false, false},
      {4, 64, 16, 4, 64, 1, false, false},
      {32, 64, 4, 32, 64, 1, false, false},
      {16, 16, 16, 16, 16, 1, false, false},
      {16, 16, 4, 16, 16, 1, false, false} ,
  };
  // clang-format on
  const int64_t waveSize = 64;

  // if can't select config from above , use this config to do
  // padding kernel for example , GEMMK/block is 16 , if your gemmK is  13 , we
  // add more 3 gemmk
  InitParams universal_Parameters = {32, 64, 4};

  int64_t obtainBlockSize(InitParamsXDL &params, int64_t waveSize) {
    return waveSize * params.gemmNPerBlock * params.gemmMPerBlock /
           (params.gemmMPerWave * params.gemmNPerWave);
  }

  LogicalResult getKBlocks(ConvolutionContext &ctx, InitParamsXDL &params,
                           int64_t *nKBlocks) {
    int64_t n = ctx.dimIndexAndSize["no"].size;
    int64_t ho = ctx.dimIndexAndSize["ho"].size;
    int64_t wo = ctx.dimIndexAndSize["wo"].size;
    int64_t g = ctx.dimIndexAndSize["g"].size;
    int64_t k = ctx.dimIndexAndSize["k"].size;
    int64_t c = ctx.dimIndexAndSize["c"].size;
    int64_t y = ctx.dimIndexAndSize["y"].size;
    int64_t x = ctx.dimIndexAndSize["x"].size;

    return mlir::miopen::calculateKBlockNum(
        n, ho, wo, g, k, c, y, x, params.gemmMPerBlock, params.gemmNPerBlock,
        params.gemmKPerBlock, params.gemmKPack, ctx.num_cu, nKBlocks);
  }

  LogicalResult calculateGemmABlockCopyPerformanceParameters(
      InitParamsXDL *param, ConvolutionContext &ctx, DerivedParams &derived) {
    int64_t blockSize = obtainBlockSize(*param, waveSize);
    return calculateInputDerivedParams(param, blockSize, ctx, true, derived);
  }

  LogicalResult calculateGemmBBlockCopyPerformanceParameters(
      InitParamsXDL *param, ConvolutionContext &ctx, DerivedParams &derived) {
    int64_t blockSize = obtainBlockSize(*param, waveSize);
    return calculateInputDerivedParams(param, blockSize, ctx, false, derived);
  }

  LogicalResult calculateLdsNumberOfByte(InitParamsXDL &param,
                                         const ConvolutionContext &ctx,
                                         DerivedParams gemmADerived,
                                         DerivedParams gemmBDerived,
                                         size_t &ldsSize) {

    int64_t threadGemmDataPerRead_GemmM =
        param.gemmMPerBlock / gemmADerived.clusterLenGemmPos2;
    int64_t threadGemmDataPerRead_GemmN =
        param.gemmNPerBlock / gemmBDerived.clusterLenGemmPos2;

    const auto max_lds_align =
        lcm(gemmADerived.dstDataPerWrite, gemmBDerived.dstDataPerWrite,
            threadGemmDataPerRead_GemmM, threadGemmDataPerRead_GemmN);

    const auto a_block_space =
        param.gemmKPerBlock *
        integer_least_multiple(param.gemmMPerBlock, max_lds_align);
    const auto b_block_space =
        param.gemmKPerBlock *
        integer_least_multiple(param.gemmNPerBlock, max_lds_align);

    ldsSize = (a_block_space + b_block_space) * sizeof(float);

    if (ldsSize > 64 * 1024) {
      return failure();
    }

    return success();
  }

  LogicalResult isValidBlockwiseGemmXDLOPS(InitParamsXDL &param,
                                           ConvolutionContext &ctx,
                                           int64_t blockSize) {
    // TBD: support fp16/bf16

    auto dataType = ctx.getDataType();
    std::vector<std::tuple<int, int, int>> validWaveGemmSize;

    if (dataType.isInteger(8)) {
      // Note: we only support two reduction xdlops in i8 therefore the
      // limited selection below
      // clang-format off
      validWaveGemmSize = {
        std::make_tuple(32, 32, 2),
        std::make_tuple(16, 16, 4)};
      // clang-format on
    } else {
      // clang-format off
      validWaveGemmSize = {
        // std::make_tuple(128, 128, 1),
        std::make_tuple(128, 64, 1),
        // std::make_tuple(128, 32, 1),
        // std::make_tuple(128, 16, 1),
        std::make_tuple(64, 128, 1),
        std::make_tuple(64, 64, 1),
        std::make_tuple(64, 32, 1),
        std::make_tuple(64, 16, 1),
        // std::make_tuple(32, 128, 1),
        std::make_tuple(32, 64, 1),
        std::make_tuple(32, 32, 2),
        // std::make_tuple(16, 128, 1),
        std::make_tuple(16, 64, 1),
        std::make_tuple(16, 16, 4),
        // std::make_tuple(8, 128, 1),
        std::make_tuple(8, 64, 1),
        // std::make_tuple(4, 128, 1),
        std::make_tuple(4, 64, 1)};
      // clang-format on
    }

    if (!std::any_of(
            validWaveGemmSize.cbegin(),
            validWaveGemmSize.cend(), [param](const auto it) noexcept->bool {
              int validMPerWave, validNPerWave, validKPerWave;
              std::tie(validMPerWave, validNPerWave, validKPerWave) = it;
              return (param.gemmMPerWave == validMPerWave) &&
                     (param.gemmNPerWave == validNPerWave) &&
                     (param.gemmKPerBlock % validKPerWave == 0);
            }))
      return failure();

    // fail with blockSize >= 512
    /// \todo fix the issue with blockSize >= 512
    if (blockSize < 64 || blockSize > 256)
      return failure();

    if ((param.gemmMPerBlock % param.gemmMPerWave) != 0)
      return failure();

    if ((param.gemmNPerBlock % param.gemmNPerWave) != 0)
      return failure();

    // TODO Remove. Note KPerBlock and KPack are independent tuning parameters.
    // There's no need to check if they are divide exactly
    if ((param.gemmKPerBlock % param.gemmKPack) != 0)
      return failure();

    // Reject invalid KPACK values.
    // For fp32: reject anything wider than 4.
    // For fp16/bf16: reject anything narrower than 4, or greater than 8.
    if (dataType.isF32() && param.gemmKPack > 4) {
      llvm::errs() << "Invalid KPACK tuning parameter: " << param.gemmKPack
                   << "\n";
      return failure();
    } else if ((dataType.isF16() || dataType.isBF16()) &&
               (param.gemmKPack != 1) &&
               ((param.gemmKPack < 4) || (param.gemmKPack > 8))) {
      llvm::errs() << "Invalid KPACK tuning parameter: " << param.gemmKPack
                   << "\n";
      return failure();
    }

    // XXX FIXME: Temporarily reject KPACK=4 for fp32 backward weight
    // convolution. It has been verified some configs would cause intermittent
    // failures.
    // TODO(whchung): Get to the bottom of this.
    if ((param.gemmKPack == 4) &&
        (ctx.getOpType() == miopen::ConvOpType::BwdWeight) &&
        dataType.isF32()) {
      llvm::errs() << "Invalid config: fp32 XDLOPS backward weight convolution "
                      "with KPACK=4\n";
      return failure();
    }

    // TODO remove : Ignore KReduction XDLOPS path for forward and backward
    // weight convolution now (unless int8). These M/NPerBlock combinations used
    // to result in lowering errors at tuning. Once non-int8 types are tested,
    // we should get rid of this limitation.
    if (param.gemmKPack > 1 && !dataType.isInteger(8)) {
      if ((param.gemmMPerBlock == 16 || param.gemmMPerBlock == 32 ||
           param.gemmMPerBlock == 64) &&
          (param.gemmNPerBlock == 16 || param.gemmNPerBlock == 32 ||
           param.gemmNPerBlock == 64)) {
        return failure();
      }

      if (param.gemmMPerBlock == 32 && param.gemmNPerBlock == 128) {
        return failure();
      }
      if (param.gemmMPerBlock == 128 && param.gemmNPerBlock == 32) {
        return failure();
      }
    }

    return success();
  }

  LogicalResult populateDerived(ConvolutionContext &ctx,
                                InitParamsXDL &validParams, GemmSize &gemmSize,
                                DerivedParams &gemmADerivedParam,
                                DerivedParams &gemmBDerivedParam,
                                DerivedOutParams &gemmCDerivedParam,
                                int64_t &blockSize, int64_t &gridSize);

  LogicalResult populatePaddingKernelDerived(
      ConvolutionContext &ctx, InitParamsXDL &validParams, GemmSize &gemmSize,
      DerivedParams &gemmADerivedParam, DerivedParams &gemmBDerivedParam,
      DerivedOutParams &gemmCDerivedParam, int64_t &blockSize,
      int64_t &gridSize);

  LogicalResult isValidGridGemmXdlops(GemmSize &gemmSize) {
    auto gemmM = gemmSize.gemmM;
    auto gemmN = gemmSize.gemmN;
    auto gemmK = gemmSize.gemmK;

    // unsupported xdlops-gemm
    if (gemmM % 16 != 0 && gemmN % 64 != 0)
      return failure();

    if ((gemmM * gemmN) % 256 == 0 && (gemmK * gemmM) % waveSize == 0 &&
        (gemmK * gemmN) % waveSize == 0 && gemmN % 16 == 0 && gemmM % 4 == 0 &&
        gemmK % 4 == 0)
      return success();
    return failure();
  }

public:
  LogicalResult obtainTuningParameters(Operation *op, int64_t blockSizeOverride,
                                       const std::string &perfConfig,
                                       InitParamsXDL &validParams,
                                       DerivedParams &gemmADerivedParam,
                                       DerivedParams &gemmBDerivedParam,
                                       DerivedOutParams &gemmCDerivedParam,
                                       int64_t &blockSize, int64_t &gridSize);

  llvm::SmallVector<InitParamsXDL, 4>
  getTuningParameters(miopen::ConvOpType dir, mlir::Type dataType) {
    if (dataType.isInteger(8)) {
      return initParametersForwardI8;
    }

    // XXX FIXME: Temporarily use another vector of heuristic tuning
    // parameters to get around the issue that when KPACK=4, fp32 backward
    // weight kernels would fail intermittently.
    // TODO(whchung): Get to the bottom of this.
    if ((dir == miopen::ConvOpType::BwdWeight) && dataType.isF32()) {
      return initParametersBwdWeightF32;
    }

    return initParameters;
  }

  InitParams getUniversalParameters() { return universal_Parameters; }

  LogicalResult isValidGemm(InitParamsXDL *param, GemmSize &gemmSize) {
    if (!(gemmSize.gemmM % param->gemmMPerBlock == 0 &&
          gemmSize.gemmN % param->gemmNPerBlock == 0 &&
          gemmSize.gemmK % (param->gemmKPerBlock * param->gemmKPack) == 0)) {
      return mlir::failure();
    }
    return mlir::success();
  }
};

// The function is used to compute extra padding sizes.
// For example, if gemmM size is 3 and gemmMPerBlock is 64,
// we set gemmMExtra be 64 so (gemmM+gemmMExtra)%gemmMPerBlock=0.
//
// Returns:
// - isOriginalKernelSupport : a bool only used in backward data convolution.
// - needExtraPad : a bool to indicate whether padding kernel is needed.
// - gemmM/N/KExtra : additional padding required along Gemm M/N/K dimension.
//                    They would be all be 0 in case needExtraPad is false.
template <typename T>
std::tuple<bool, bool, int64_t, int64_t, int64_t>
calculatePaddingKernelSize(int64_t gemmMSize, int64_t gemmNSize,
                           int64_t gemmKSize, miopen::ConvOpType dir,
                           mlir::Type dataType, T populateParams) {
  bool isOriginalKernelSupport = true;
  bool needExtraPad = false;
  int64_t gemmMExtra, gemmNExtra, gemmKExtra;
  gemmMExtra = gemmNExtra = gemmKExtra = 0;

  auto configParams = populateParams.getTuningParameters(dir, dataType);
  size_t numOfFailedConfigs = 0;
  for (auto &params : configParams) {
    if (gemmMSize % params.gemmMPerBlock == 0 &&
        gemmKSize % params.gemmKPerBlock == 0 &&
        gemmNSize % params.gemmNPerBlock == 0) {
      isOriginalKernelSupport = true;
      break;
    }
    isOriginalKernelSupport = false;
    numOfFailedConfigs++;
  }

  auto extraParams = populateParams.getUniversalParameters();
  if (numOfFailedConfigs == configParams.size()) {
    needExtraPad = true;
    int64_t gemmMRemain, gemmKRemain, gemmNRemain;

    gemmMRemain = gemmMSize % extraParams.gemmMPerBlock;
    if (gemmMRemain != 0)
      gemmMExtra = extraParams.gemmMPerBlock - gemmMRemain;

    gemmNRemain = gemmNSize % extraParams.gemmNPerBlock;
    if (gemmNRemain != 0)
      gemmNExtra = extraParams.gemmNPerBlock - gemmNRemain;

    gemmKRemain = gemmKSize % extraParams.gemmKPerBlock;
    if (gemmKRemain != 0)
      gemmKExtra = extraParams.gemmKPerBlock - gemmKRemain;

    // llvm::errs() << "gemmMExtra: " << gemmMExtra << "gemmNExtra: " <<
    // gemmNExtra << "gemmKExtra: " << gemmKExtra << "\n";
  }

  return std::make_tuple(isOriginalKernelSupport, needExtraPad, gemmMExtra,
                         gemmNExtra, gemmKExtra);
}

#endif // MLIR_DIALECT_MIOPEN_GRIDWISE_GEMM_PARAMS_H
