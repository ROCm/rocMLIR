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

#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Tuning/ConvContext.h"
#include "mlir/Dialect/MIOpen/Tuning/Serializable.h"
#include "mlir/Dialect/MIOpen/utility/math.hpp"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"

#include <string>
#include <unordered_map>

using namespace mlir;

LLVM_YAML_IS_STRING_MAP(int)

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
  int64_t dstVectorWriteDim;
  int64_t srcDataPerRead;
  int64_t dstDataPerWrite;
  int64_t clusterLenGemmPos0; // G
  int64_t clusterLenGemmPos1;
  int64_t clusterLenGemmPos2;
  DerivedParams()
      : srcVectorReadDim(0), dstVectorWriteDim(0), srcDataPerRead(1),
        dstDataPerWrite(1), clusterLenGemmPos1(0), clusterLenGemmPos2(0) {}
};

class PopulateParamsBase {
public:
  static void obtainGemmADimKVectorizable(
      mlir::miopen::ConvOpType opType,
      llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal,
      bool &input1GemmKVectorizable) {
    // Vectorizable flag is opposite between forwad and bwd_data
    if (opType == mlir::miopen::ConvOpType::Conv2DOpType) {
      // When K is not the fastest changing dimension,
      // gemmK dimension is vectorizable, gemmM is not, and vice versa.
      // Vectorization width depending on which among C, Y, X be the fastest
      // changing dimension.
      if (dimIndexVal["k"].first == 4) {
        input1GemmKVectorizable = false;
      } else {
        input1GemmKVectorizable = true;
      }
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
      // always load gemmM first
      input1GemmKVectorizable = false;
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
      // When K is the fastest changing dimension,
      // gemmM dimension is vectorizable, gemmK is not, and vice versa.
      // Vectorization width depending on which among N, and HoWo be the fastest
      // changing dimension.
      if (dimIndexVal["k"].first == 4) {
        input1GemmKVectorizable = false;
      } else {
        input1GemmKVectorizable = true;
      }
    }
  }

  static void obtainGemmBDimKVectorizable(
      mlir::miopen::ConvOpType opType,
      llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal,
      bool &input2GemmKVectorizable) {
    // Vectorizable flag is opposite between forwad and bwd_data
    if (opType == mlir::miopen::ConvOpType::Conv2DOpType) {
      // For input tensor.
      // When C is the fastest changing dimension,
      // gemmK dimension is vectorizable, gemmN is not, and vice versa.
      // Vectorization width depending on length of C.
      if (dimIndexVal["ci"].first == 4) {
        input2GemmKVectorizable = true;
      } else {
        input2GemmKVectorizable = false;
      }
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
      // For output tensor.
      // When K is the fastest changing dimension(3),
      // gemmK dimension is vectorizable, gemmN is not, and vice versa.
      // Vectorization width depending on length of K.
      if (dimIndexVal["ko"].first == 4) {
        input2GemmKVectorizable = true;
      } else {
        input2GemmKVectorizable = false;
      }
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
      // For input tensor
      // When C is the fastest changing dimension,
      // gemmN dimension is vectorizable, gemmK is not, and vice versa.
      // Vectorization width depending on length of C.
      if (dimIndexVal["ci"].first == 4) {
        input2GemmKVectorizable = false;
      } else {
        input2GemmKVectorizable = true;
      }
    }
  }

  static void obtainFilterVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto dimIndexVal = ctx.dimIndexVal;
    // Vectorization length logic is the same for forward and bwd_data
    if (dimIndexVal["k"].first == 4) {
      vecLen = dimIndexVal["k"].second;
    } else if (dimIndexVal["k"].first == 1) {
      // dimKF is the lowest changing dimension, which means dimC/dimY/dimX
      vecLen = dimIndexVal["c"].second * dimIndexVal["y"].second *
               dimIndexVal["x"].second;
    } else if (dimIndexVal["k"].first == 2) {
      // K's position is at 2, vectorization legnth is last two dimension
      if (dimIndexVal["c"].first == 1) {
        vecLen = dimIndexVal["y"].second * dimIndexVal["x"].second;
      } else if (dimIndexVal["y"].first == 1) {
        vecLen = dimIndexVal["c"].second * dimIndexVal["x"].second;
      } else {
        vecLen = dimIndexVal["c"].second * dimIndexVal["y"].second;
      }
    } else {
      // K's position is 3, vectorization legnth is last dimension
      if (dimIndexVal["c"].first == 4) {
        vecLen = dimIndexVal["c"].second;
      } else if (dimIndexVal["y"].first == 4) {
        vecLen = dimIndexVal["y"].second;
      } else {
        vecLen = dimIndexVal["x"].second;
      }
    }
  }

  static void obtainBwdDataFilterVecLen(ConvolutionContext &ctx,
                                        int64_t &vecLen) {
    auto dimIndexVal = ctx.dimIndexVal;
    // Vectorization length logic is the same for forward and bwd_data
    if (dimIndexVal["c"].first == 4) {
      vecLen = dimIndexVal["c"].second;
    } else if (dimIndexVal["c"].first == 2) {
      // C's position is at 2, vectorization legnth depend last two dimension
      if (dimIndexVal["y"].second == 1 && dimIndexVal["x"].second == 1) {
        vecLen = dimIndexVal["c"].second;
      } else {
        vecLen = 1;
      }
    } else {
      vecLen = 1;
    }
  }
  static void obtainInputVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto dimIndexVal = ctx.dimIndexVal;
    if (dimIndexVal["ni"].first == 4) {
      vecLen = dimIndexVal["ni"].second;
    } else if (dimIndexVal["ci"].first == 4) {
      vecLen = dimIndexVal["ci"].second;
    } else {
      if (dimIndexVal["x"].second == 1 && dimIndexVal["y"].second == 1 &&
          ctx.strideVal[0] == 1 && ctx.strideVal[1] == 1 &&
          ctx.paddingVal[0] == 0 && ctx.paddingVal[1] == 0 &&
          ctx.paddingVal[2] == 0 && ctx.paddingVal[3] == 0)
        vecLen = dimIndexVal["ho"].second * dimIndexVal["wo"].second;
      else
        vecLen = 1;
    }
  }
  static void obtainBwdDataOutputVecLen(ConvolutionContext &ctx,
                                        int64_t &vecLen) {
    auto dimIndexVal = ctx.dimIndexVal;
    if (dimIndexVal["ko"].first == 4) {
      vecLen = dimIndexVal["ko"].second;
    } else if (dimIndexVal["no"].first == 4) {
      vecLen = dimIndexVal["no"].second;
    } else if (dimIndexVal["no"].first == 0) {
      if (dimIndexVal["ho"].first == 3 && dimIndexVal["wo"].first == 4) {
        if (dimIndexVal["y"].second == 1 && dimIndexVal["x"].second == 1)
          vecLen = dimIndexVal["ho"].second * dimIndexVal["wo"].second;
        else
          vecLen = 1;
      } else
        vecLen = 1;
    } else {
      vecLen = 1;
    }
  }

  static void obtainOutputVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto dimIndexVal = ctx.dimIndexVal;
    if (dimIndexVal["ko"].first == 4) {
      vecLen = dimIndexVal["ko"].second;
    } else if (dimIndexVal["ko"].first == 1) {
      // dimKO is the lowest changing dimension, which means dimN/dimHo/dimWo
      vecLen = dimIndexVal["no"].second * dimIndexVal["ho"].second *
               dimIndexVal["wo"].second;
    } else if (dimIndexVal["ko"].first == 2) {
      // Ko's position is at 2, vectorization legnth is last two dimensions
      if (dimIndexVal["no"].first == 1) {
        vecLen = dimIndexVal["ho"].second * dimIndexVal["wo"].second;
      } else if (dimIndexVal["ho"].first == 1) {
        vecLen = dimIndexVal["no"].second * dimIndexVal["wo"].second;
      } else {
        vecLen = dimIndexVal["no"].second * dimIndexVal["ho"].second;
      }
    } else {
      // K's position is 3, vectorization legnth is last dimension
      if (dimIndexVal["no"].first == 4) {
        vecLen = dimIndexVal["no"].second;
      } else if (dimIndexVal["ho"].first == 4) {
        vecLen = dimIndexVal["ho"].second;
      } else {
        vecLen = dimIndexVal["wo"].second;
      }
    }
  }

  static void obtainGemmAVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto opType = ctx.opType;
    if (opType == mlir::miopen::ConvOpType::Conv2DOpType) {
      obtainFilterVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
      obtainBwdDataFilterVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
      obtainOutputVecLen(ctx, vecLen);
    }
  }

  static void obtainGemmBVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto opType = ctx.opType;
    if (opType == mlir::miopen::ConvOpType::Conv2DOpType) {
      obtainInputVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
      obtainBwdDataOutputVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
      obtainInputVecLen(ctx, vecLen);
    }
  }

  static void obtainGemmCVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
    auto opType = ctx.opType;
    if (opType == mlir::miopen::ConvOpType::Conv2DOpType) {
      obtainOutputVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
      obtainInputVecLen(ctx, vecLen);
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
      obtainFilterVecLen(ctx, vecLen);
    }
  }

protected:
  mlir::LogicalResult calculateInputDerivedParams(InitParams *param,
                                                  int64_t blockSize,
                                                  ConvolutionContext &ctx,
                                                  bool isGemmA,
                                                  DerivedParams &derived) {

    bool gemmPos1Vectorizable = false;
    int64_t vectorizableLength = 0;
    if (isGemmA) {
      obtainGemmADimKVectorizable(ctx.opType, ctx.dimIndexVal,
                                  gemmPos1Vectorizable);
      obtainGemmAVecLen(ctx, vectorizableLength);
    } else {
      obtainGemmBDimKVectorizable(ctx.opType, ctx.dimIndexVal,
                                  gemmPos1Vectorizable);
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

    // srcDataPerRead bounded by size of threadwise copy
    const int64_t vectorizationSize = 4;
    if ((vectorizableLength > 0) && (vectorizableLength % 4 == 0)) {
      derived.srcDataPerRead = gcd(vectorizationSize, dataPerThreadCopy);
    }

    // decide threadwise copy lengths
    const auto dataPerThreadCopyGemmVectorized = derived.srcDataPerRead;
    const auto dataPerThreadCopyGemmNonvectorized =
        dataPerThreadCopy / dataPerThreadCopyGemmVectorized;

    int64_t dataPerThreadCopyGemmPos1 = 0;
    int64_t dataPerThreadCopyGemmPos2 = 0;
    if (gemmPos1Vectorizable) {
      dataPerThreadCopyGemmPos1 = dataPerThreadCopyGemmVectorized;
      dataPerThreadCopyGemmPos2 = dataPerThreadCopyGemmNonvectorized;
      derived.srcVectorReadDim = 1;
    } else {
      dataPerThreadCopyGemmPos1 = dataPerThreadCopyGemmNonvectorized;
      dataPerThreadCopyGemmPos2 = dataPerThreadCopyGemmVectorized;
      derived.srcVectorReadDim = 2;
    }

    // dstDataPerWrite also bounded by size of threadwise copy
    derived.dstDataPerWrite = gcd(vectorizationSize, dataPerThreadCopyGemmPos2);

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

  static void obtainGemmSize(ConvolutionContext &ctx, GemmSize &gemmSize) {
    gemmSize.gemmG = ctx.dimIndexVal["g"].second;

    if (ctx.opType == mlir::miopen::ConvOpType::Conv2DOpType) {
      gemmSize.gemmM = ctx.dimIndexVal["k"].second;
      gemmSize.gemmN = ctx.dimIndexVal["no"].second *
                       ctx.dimIndexVal["ho"].second *
                       ctx.dimIndexVal["wo"].second;
      gemmSize.gemmK = ctx.dimIndexVal["c"].second *
                       ctx.dimIndexVal["y"].second *
                       ctx.dimIndexVal["x"].second;
    } else if (ctx.opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
      int64_t y, x, ho, wo, hi, wi;
      y = x = ho = wo = hi = wi = 0;
      y = ctx.dimIndexVal["y"].second;
      x = ctx.dimIndexVal["x"].second;
      ho = ctx.dimIndexVal["ho"].second;
      wo = ctx.dimIndexVal["wo"].second;
      hi = ctx.dimIndexVal["hi"].second;
      wi = ctx.dimIndexVal["wi"].second;
      auto strideH = ctx.strideVal[0];
      auto strideW = ctx.strideVal[1];
      auto dilationH = ctx.dilationVal[0];
      auto dilationW = ctx.dilationVal[1];
      auto leftPadH = ctx.paddingVal[0];
      auto leftPadW = ctx.paddingVal[1];

      auto gcdStrideDilationH = math::gcd(strideH, dilationH);
      auto gcdStrideDilationW = math::gcd(strideW, dilationW);

      auto yTilda = strideH / gcdStrideDilationH;
      auto xTilda = strideW / gcdStrideDilationW;

      auto hTilda =
          ho + math::integer_divide_ceil(dilationH * (y - 1), strideH);
      auto wTilda =
          wo + math::integer_divide_ceil(dilationW * (x - 1), strideW);

      auto iHTildaLeft = math::integer_divide_floor(
          std::max(0l, leftPadH - dilationH * (yTilda - 1)), strideH);
      auto iWTildaLeft = math::integer_divide_floor(
          std::max(0l, leftPadW - dilationW * (xTilda - 1)), strideW);

      auto iHTildaRight = std::min(
          hTilda, math::integer_divide_ceil(leftPadH + hi - 1, strideH) + 1);
      auto iWTildaRight = std::min(
          wTilda, math::integer_divide_ceil(leftPadW + wi - 1, strideW) + 1);

      auto hTildaSlice = iHTildaRight - iHTildaLeft;
      auto wTildaSlice = iWTildaRight - iWTildaLeft;

      auto gemmId = ctx.gemmId;
      auto iYTilda = gemmId / xTilda;
      auto iXTilda = gemmId % xTilda;
      auto yDotSlice = math::integer_divide_ceil(y - iYTilda, yTilda);
      auto xDotSlice = math::integer_divide_ceil(x - iXTilda, xTilda);

      gemmSize.gemmM = ctx.dimIndexVal["c"].second;
      gemmSize.gemmN = ctx.dimIndexVal["no"].second * hTildaSlice * wTildaSlice;
      gemmSize.gemmK = ctx.dimIndexVal["k"].second * yDotSlice * xDotSlice;
    } else if (ctx.opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
      gemmSize.gemmM = ctx.dimIndexVal["k"].second;
      gemmSize.gemmK = ctx.dimIndexVal["no"].second *
                       ctx.dimIndexVal["ho"].second *
                       ctx.dimIndexVal["wo"].second;
      gemmSize.gemmN = ctx.dimIndexVal["c"].second *
                       ctx.dimIndexVal["y"].second *
                       ctx.dimIndexVal["x"].second;
    }
  }

  int64_t obtainGridSize(GemmSize &gemmSize, InitParams *param) {
    return (gemmSize.gemmM / param->gemmMPerBlock) *
           (gemmSize.gemmN / param->gemmNPerBlock) * gemmSize.gemmG;
  }

  mlir::LogicalResult isValidGemm(InitParams *param, GemmSize &gemmSize) {
    if (!(gemmSize.gemmM % param->gemmMPerBlock == 0 &&
          gemmSize.gemmN % param->gemmNPerBlock == 0 &&
          gemmSize.gemmK % param->gemmKPerBlock == 0)) {
      return mlir::failure();
    }
    return mlir::success();
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

  int64_t calculateGemmCDestDataPerWrite(const InitParamsNonXDL &param,
                                         ConvolutionContext &ctx) {
    int64_t outputVecLen = 0;
    if ((ctx.opType == miopen::ConvOpType::Conv2DOpType) &&
        (ctx.dimIndexVal["ko"].first == 4)) {
      // gemmM vectorizable. However, there is no parameters for vectorizing
      // gemmM dimension for matrix C. Do nothing here.
    } else if ((ctx.opType == miopen::ConvOpType::Conv2DBwdDataOpType) &&
               (ctx.dimIndexVal["ci"].first == 4)) {
      // gemmM vectorizable. However, there is no parameters for vectorizing
      // gemmM dimension for matrix C. Do nothing here.
    } else {
      obtainGemmCVecLen(ctx, outputVecLen);
    }

    outputVecLen = gcd(outputVecLen, param.gemmNPerThread);

    if ((outputVecLen > 0) && (outputVecLen % 4 == 0)) {
      return 4;
    } else if ((outputVecLen > 0) && (outputVecLen % 2 == 0)) {
      return 2;
    }

    return 1;
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

  LogicalResult populateDerived(ConvolutionContext &ctx,
                                InitParamsNonXDL &validParams,
                                GemmSize &gemmSize,
                                DerivedParams &gemmADerivedParam,
                                DerivedParams &gemmBDerivedParam,
                                DerivedBlockGemmParams &blockGemmDerivedParam,
                                int64_t &gemmCDstPerWrite, int64_t &gridSize);

  LogicalResult populatePaddingKernelDerived(
      ConvolutionContext &ctx, InitParamsNonXDL &validParams,
      GemmSize &gemmSize, DerivedParams &gemmADerivedParam,
      DerivedParams &gemmBDerivedParam,
      DerivedBlockGemmParams &blockGemmDerivedParam, int64_t &gemmCDstPerWrite,
      int64_t &gridSize);

public:
  LogicalResult paramsFromCtx(ConvolutionContext &ctx,
                              int64_t blockSizeOverride,
                              InitParamsNonXDL &validParams,
                              DerivedParams &gemmADerivedParam,
                              DerivedParams &gemmBDerivedParam,
                              DerivedBlockGemmParams &blockGemmDerivedParam,
                              int64_t &gemmCDstPerWrite, int64_t &gridSize);

  llvm::SmallVector<InitParamsNonXDL, 8> getTuningParameters() {
    return initParameters;
  }

  InitParams getUniversalParameters() { return universal_Parameters; }
};

class PopulateParamsXDL : public PopulateParamsBase {
private:
  llvm::SmallVector<InitParamsXDL, 4> initParameters = {
      // M/block N/block K/block M/wave N/wave kPack aCopyMore bCopyMore
      {256, 128, 16, 128, 64, 0, false, false},
      {128, 128, 16, 64, 64, 0, false, false},
      {8, 64, 8, 8, 64, 0, false, false},
      {4, 64, 16, 4, 64, 0, false, false},
      {32, 64, 4, 32, 64, 0, false, false},
      {16, 16, 16, 16, 16, 0, false, false},
      {16, 16, 4, 16, 16, 0, false, false},
  };
  const int64_t waveSize = 64;

  // if can't select config from above , use this config to do
  // padding kernel for example , GEMMK/block is 16 , if your gemmK is  13 , we
  // add more 3 gemmk
  InitParams universal_Parameters = {32, 64, 4};

  int64_t obtainBlockSize(InitParamsXDL &params, int64_t waveSize) {
    return waveSize * params.gemmNPerBlock * params.gemmMPerBlock /
           (params.gemmMPerWave * params.gemmNPerWave);
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

  LogicalResult isValidblockwisegemmxdlops(InitParamsXDL &param,
                                           int64_t blockSize) {
    // TBD: support fp16/bf16

    std::vector<std::tuple<int, int, int>> validWaveGemmSize = {
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
                     [param](const auto it) noexcept -> bool {
                       int validMPerWave, validNPerWave, validKPerWave;
                       std::tie(validMPerWave, validNPerWave, validKPerWave) =
                           it;
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

    return success();
  }

  LogicalResult populateDerived(ConvolutionContext &ctx,
                                InitParamsXDL &validParams, GemmSize &gemmSize,
                                DerivedParams &gemmADerivedParam,
                                DerivedParams &gemmBDerivedParam,
                                int64_t &blockSize, int64_t &gridSize);

  LogicalResult populatePaddingKernelDerived(
      ConvolutionContext &ctx, InitParamsXDL &validParams, GemmSize &gemmSize,
      DerivedParams &gemmADerivedParam, DerivedParams &gemmBDerivedParam,
      int64_t &blockSize, int64_t &gridSize);

public:
  LogicalResult paramsFromCtx(ConvolutionContext &ctx,
                              int64_t blockSizeOverride,
                              InitParamsXDL &validParams,
                              DerivedParams &gemmADerivedParam,
                              DerivedParams &gemmBDerivedParam,
                              int64_t &blockSize, int64_t &gridSize);

  llvm::SmallVector<InitParamsXDL, 4> getTuningParameters() {
    return initParameters;
  }

  InitParams getUniversalParameters() { return universal_Parameters; }
};

#endif // MLIR_DIALECT_MIOPEN_GRIDWISE_GEMM_PARAMS_H
