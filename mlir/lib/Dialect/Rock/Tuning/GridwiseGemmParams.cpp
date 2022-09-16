#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Rock.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/GemmContext.h"
#include "mlir/Dialect/Rock/Tuning/SqliteDb.h"
#include "mlir/Dialect/Rock/utility/math.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

llvm::raw_ostream &mlir::rock::operator<<(llvm::raw_ostream &os,
                                            GemmDimension dim) {
  switch (dim) {
  case GemmDimension::G:
    return os << "GemmDimmension::G";
  case GemmDimension::K:
    return os << "GemmDimension::K";
  case GemmDimension::MorN:
    return os << "GemmDimension::MorN";
  }
}

static void
obtainGemmADimKVectorizable(ConvOpType opType,
                            llvm::StringMap<DimIndexAndSize> &dimIndexAndSize,
                            bool &input1GemmKVectorizable) {
  // Vectorizable flag is opposite between forwad and bwd_data
  if (opType == ConvOpType::Fwd) {
    // When K is not the fastest changing dimension,
    // gemmK dimension is vectorizable, gemmM is not, and vice versa.
    // Vectorization width depending on which among C, Y, X be the fastest
    // changing dimension.
    if (dimIndexAndSize["k"].index == 4) {
      input1GemmKVectorizable = false;
    } else {
      input1GemmKVectorizable = true;
    }
  } else if (opType == ConvOpType::BwdData) {
    // always load gemmM first
    input1GemmKVectorizable = false;
  } else if (opType == ConvOpType::BwdWeight) {
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
obtainGemmBDimKVectorizable(ConvOpType opType,
                            llvm::StringMap<DimIndexAndSize> &dimIndexAndSize,
                            bool &input2GemmKVectorizable) {
  // Vectorizable flag is opposite between forwad and bwd_data
  if (opType == ConvOpType::Fwd) {
    // For input tensor.
    // When C is the fastest changing dimension,
    // gemmK dimension is vectorizable, gemmN is not, and vice versa.
    // Vectorization width depending on length of C.
    if (dimIndexAndSize["ci"].index == 4) {
      input2GemmKVectorizable = true;
    } else {
      input2GemmKVectorizable = false;
    }
  } else if (opType == ConvOpType::BwdData) {
    // For output tensor.
    // When K is the fastest changing dimension(3),
    // gemmK dimension is vectorizable, gemmN is not, and vice versa.
    // Vectorization width depending on length of K.
    if (dimIndexAndSize["ko"].index == 4) {
      input2GemmKVectorizable = true;
    } else {
      input2GemmKVectorizable = false;
    }
  } else if (opType == ConvOpType::BwdWeight) {
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
    if (dimIndexAndSize["ho"].index == 3 && dimIndexAndSize["wo"].index == 4) {
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
  if (opType == ConvOpType::Fwd) {
    obtainFilterVecLen(ctx, vecLen);
  } else if (opType == ConvOpType::BwdData) {
    obtainBwdDataFilterVecLen(ctx, vecLen);
  } else if (opType == ConvOpType::BwdWeight) {
    obtainOutputVecLen(ctx, vecLen);
  }
}

static void obtainGemmBVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
  auto opType = ctx.opType;
  if (opType == ConvOpType::Fwd) {
    obtainInputVecLen(ctx, vecLen);
  } else if (opType == ConvOpType::BwdData) {
    obtainBwdDataOutputVecLen(ctx, vecLen);
  } else if (opType == ConvOpType::BwdWeight) {
    obtainInputVecLen(ctx, vecLen);
  }
}

static void obtainGemmCVecLen(ConvolutionContext &ctx, int64_t &vecLen) {
  auto opType = ctx.opType;
  if (opType == ConvOpType::Fwd) {
    obtainOutputVecLen(ctx, vecLen);
  } else if (opType == ConvOpType::BwdData) {
    obtainInputVecLen(ctx, vecLen);
  } else if (opType == ConvOpType::BwdWeight) {
    obtainFilterVecLen(ctx, vecLen);
  }
}

LogicalResult calculateInputDerivedParams(const InitParams &param,
                                          int64_t blockSize,
                                          ConvolutionContext &ctx, bool isGemmA,
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
    dataPerThreadCopy = (param.gemmKPerBlock * param.gemmMPerBlock) / blockSize;
  } else {
    dataPerThreadCopy = (param.gemmKPerBlock * param.gemmNPerBlock) / blockSize;
  }

  if (!(dataPerThreadCopy > 0))
    return failure();

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
  if (opType == ConvOpType::BwdData || opType == ConvOpType::BwdWeight) {
    vectorizationSize = 1;
  }

  // srcDataPerRead bounded by size of threadwise copy
  if ((vectorizableLength > 0) && (vectorizableLength % 4 == 0)) {
    derived.srcDataPerRead =
        math_util::gcd(vectorizationSize, dataPerThreadCopy);
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

  // calculate blockwise copy thread cluster lengths
  if (isGemmA) {
    derived.clusterLenGemmPos1 =
        param.gemmKPerBlock / dataPerThreadCopyGemmPos1;
    derived.clusterLenGemmPos2 =
        param.gemmMPerBlock / dataPerThreadCopyGemmPos2;
  } else {
    derived.clusterLenGemmPos1 =
        param.gemmKPerBlock / dataPerThreadCopyGemmPos1;
    derived.clusterLenGemmPos2 =
        param.gemmNPerBlock / dataPerThreadCopyGemmPos2;
  }
  if (!(derived.clusterLenGemmPos1 > 0 && derived.clusterLenGemmPos2 > 0))
    return failure();

  return success();
}

LogicalResult calculateOutputDerivedParams(const InitParams &params,
                                           int64_t blockSize,
                                           ConvolutionContext &ctx,
                                           DerivedOutParams &out) {
  int64_t cVectorLength = 0;
  ConvOpType op = ctx.getOpType();

  obtainGemmCVecLen(ctx, cVectorLength);
  int64_t dataPerThread =
      (params.gemmMPerBlock * params.gemmNPerBlock) / blockSize;
  if (!(dataPerThread > 0)) {
    return failure();
  }

  // TODO: Allow vectorization group size of 2
  int64_t vectorizationSize = 4;
  // No swizzling or vectorization for backward data
  // TODO(kdrewnia): Understand when it might be possible
  if (ConvOpType::BwdData == op) {
    vectorizationSize = 1;
  }

  if ((cVectorLength > 0) && (dataPerThread % vectorizationSize == 0) &&
      (cVectorLength % vectorizationSize == 0)) {
    out.dataPerCopy = math_util::gcd(dataPerThread, vectorizationSize);
  } else {
    out.dataPerCopy = 1;
  }

  auto &dimIndexAndSize = ctx.dimIndexAndSize;
  // Find dimensions in which the copy will take place
  switch (op) {
  case ConvOpType::Fwd:
    if (dimIndexAndSize["ko"].index == 4) {
      out.gemmVectorDim = gemmCDimM;
      out.destVectorDim = 4;
    } else {
      out.gemmVectorDim = gemmCDimN;
      // This relies on assumptions about how we load our data for GEMM
      out.destVectorDim = dimIndexAndSize["wo"].index;
    }
    break;
  case ConvOpType::BwdWeight:
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
  case ConvOpType::BwdData:
    out.gemmVectorDim = -1;
    out.destVectorDim = -1;
    break;
  }
  return success();
}

static void obtainGemmSize(ConvolutionContext &ctx, GemmSize &gemmSize) {
  gemmSize.gemmG = ctx.dimIndexAndSize["g"].size;

  if (ctx.opType == ConvOpType::Fwd) {
    gemmSize.gemmM = ctx.dimIndexAndSize["k"].size;
    gemmSize.gemmN = ctx.dimIndexAndSize["no"].size *
                     ctx.dimIndexAndSize["ho"].size *
                     ctx.dimIndexAndSize["wo"].size;
    gemmSize.gemmK = ctx.dimIndexAndSize["c"].size *
                     ctx.dimIndexAndSize["y"].size *
                     ctx.dimIndexAndSize["x"].size;
  } else if (ctx.opType == ConvOpType::BwdData) {
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
    auto leftPadW = ctx.paddingVal[2];

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
    gemmSize.gemmN = ctx.dimIndexAndSize["no"].size * hTildaSlice * wTildaSlice;
    gemmSize.gemmK = ctx.dimIndexAndSize["k"].size * yDotSlice * xDotSlice;
  } else if (ctx.opType == ConvOpType::BwdWeight) {
    gemmSize.gemmM = ctx.dimIndexAndSize["k"].size;
    gemmSize.gemmK = ctx.dimIndexAndSize["no"].size *
                     ctx.dimIndexAndSize["ho"].size *
                     ctx.dimIndexAndSize["wo"].size;
    gemmSize.gemmN = ctx.dimIndexAndSize["c"].size *
                     ctx.dimIndexAndSize["y"].size *
                     ctx.dimIndexAndSize["x"].size;
  }
}

int64_t obtainGridSize(GemmSize &gemmSize, const InitParams &param) {
  return (gemmSize.gemmM / param.gemmMPerBlock) *
         (gemmSize.gemmN / param.gemmNPerBlock) * gemmSize.gemmG;
}

/// Non-xdlops
// clang-format off
const InitParamsNonXDL
PopulateParams::initParameters[PopulateParams::nInitParameters] = {
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

const InitParams PopulateParams::universalParameters = {64, 64, 16};

LogicalResult PopulateParams::calculateGemmABlockCopyPerformanceParameters(
    const InitParamsNonXDL &param, ConvolutionContext &ctx,
    DerivedParams &derived) {
  return calculateInputDerivedParams(param, param.blockSize, ctx, true,
                                     derived);
}

LogicalResult PopulateParams::calculateGemmBBlockCopyPerformanceParameters(
    const InitParamsNonXDL &param, ConvolutionContext &ctx,
    DerivedParams &derived) {

  return calculateInputDerivedParams(param, param.blockSize, ctx, false,
                                     derived);
}

LogicalResult PopulateParams::calculateGemmCBlockwiseCopyParams(
    const InitParamsNonXDL &params, ConvolutionContext &ctx,
    DerivedOutParams &out) {
  return calculateOutputDerivedParams(params, params.blockSize, ctx, out);
}

LogicalResult PopulateParams::calculateBlockGemmPerformanceParameters(
    const InitParamsNonXDL &param, const ConvolutionContext &ctx,
    DerivedBlockGemmParams &derived) {

  derived.gemmMThreadsPerCuwave = 0;
  derived.gemmNThreadsPerCuwave = 0;
  derived.gemmMCuwavesPerBlock = 0;
  derived.gemmNCuwavesPerBlock = 0;

  if (param.blockSize == 64) {
    derived.gemmMThreadsPerCuwave = 4;
    derived.gemmNThreadsPerCuwave = 4;
    derived.gemmMCuwavesPerBlock = 2;
    derived.gemmNCuwavesPerBlock = 2;
  } else if (param.blockSize == 128) {
    derived.gemmMThreadsPerCuwave = 4;
    derived.gemmNThreadsPerCuwave = 4;
    derived.gemmMCuwavesPerBlock = 4;
    derived.gemmNCuwavesPerBlock = 2;
  } else if (param.blockSize == 256) {
    derived.gemmMThreadsPerCuwave = 4;
    derived.gemmNThreadsPerCuwave = 4;
    derived.gemmMCuwavesPerBlock = 4;
    derived.gemmNCuwavesPerBlock = 4;
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
      derived.gemmMThreadsPerCuwave * derived.gemmMCuwavesPerBlock;
  const auto threadGemmNPerCluster =
      derived.gemmNThreadsPerCuwave * derived.gemmNCuwavesPerBlock;

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
LogicalResult PopulateParams::populateDerived(
    ConvolutionContext &ctx, const InitParamsNonXDL &params, GemmSize &gemmSize,
    DerivedParams &gemmADerivedParam, DerivedParams &gemmBDerivedParam,
    DerivedBlockGemmParams &blockGemmDerivedParam,
    DerivedOutParams &gemmCDerivedParams, uint32_t &gridSize) {

  LogicalResult res = failure();
  res = isValidGemm(params, gemmSize);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Gemm sizes, M: " << gemmSize.gemmM
               << " N: " << gemmSize.gemmN << " K: " << gemmSize.gemmK << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Gemm size and gemm/block "
                            << "size does not divide exactly.\n");
    return failure();
  }

  if (ctx.opType == ConvOpType::BwdData &&
      !(gemmSize.gemmM % 32 == 0 && gemmSize.gemmN % 32 == 0 &&
        gemmSize.gemmK % 4 == 0)) {
    LLVM_DEBUG(llvm::dbgs() << "Invalid gemm sizes for backward data.\n");
    return failure();
  }

  res = calculateGemmABlockCopyPerformanceParameters(params, ctx,
                                                     gemmADerivedParam);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmA tuning parameter "
                            << " size.\n");
    return failure();
  }

  res = calculateGemmBBlockCopyPerformanceParameters(params, ctx,
                                                     gemmBDerivedParam);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmB tuning parameter "
                            << " size.\n");
    return failure();
  }

  res = calculateBlockGemmPerformanceParameters(params, ctx,
                                                blockGemmDerivedParam);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent blockGemm tuning parameter "
                            << " size.\n");
    return failure();
  }

  gridSize = obtainGridSize(gemmSize, params);
  res = calculateGemmCBlockwiseCopyParams(params, ctx, gemmCDerivedParams);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmC tuning parametrs.\n");
    return failure();
  }
  return success();
}

LogicalResult PopulateParams::populatePaddingKernelDerived(
    ConvolutionContext &ctx, const InitParamsNonXDL &param, GemmSize &gemmSize,
    DerivedParams &gemmADerivedParam, DerivedParams &gemmBDerivedParam,
    DerivedBlockGemmParams &blockGemmDerivedParam,
    DerivedOutParams &gemmCDerivedParam, uint32_t &gridSize) {

  LogicalResult res = failure();
  InitParams paddingParam = getUniversalParameters();

  // find complete config for paddingParam
  // if mblock,nblock,kblock is the same with this tuning parameter
  if (paddingParam.gemmMPerBlock != param.gemmMPerBlock ||
      paddingParam.gemmNPerBlock != param.gemmNPerBlock ||
      paddingParam.gemmKPerBlock != param.gemmKPerBlock)
    return failure();

  if (gemmSize.gemmM % param.gemmMPerBlock != 0)
    gemmSize.gemmM = gemmSize.gemmM + (param.gemmMPerBlock -
                                       gemmSize.gemmM % param.gemmMPerBlock);

  if (gemmSize.gemmN % param.gemmNPerBlock != 0)
    gemmSize.gemmN = gemmSize.gemmN + (param.gemmNPerBlock -
                                       gemmSize.gemmN % param.gemmNPerBlock);

  if (gemmSize.gemmK % param.gemmKPerBlock != 0)
    gemmSize.gemmK = gemmSize.gemmK + (param.gemmKPerBlock -
                                       gemmSize.gemmK % param.gemmKPerBlock);

  res = calculateGemmABlockCopyPerformanceParameters(param, ctx,
                                                     gemmADerivedParam);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmA tuning parameter "
                            << " size.\n");
    return failure();
  }

  res = calculateGemmBBlockCopyPerformanceParameters(param, ctx,
                                                     gemmBDerivedParam);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmB tuning parameter "
                            << " size.\n");
    return failure();
  }

  res = calculateBlockGemmPerformanceParameters(param, ctx,
                                                blockGemmDerivedParam);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent blockGemm tuning parameter "
                            << " size.\n");
    return failure();
  }

  gridSize = obtainGridSize(gemmSize, param);
  res = calculateGemmCBlockwiseCopyParams(param, ctx, gemmCDerivedParam);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmC tuning parametrs.\n");
    return failure();
  }
  return success();
}

LogicalResult PopulateParams::obtainTuningParameters(
    Operation *op, uint32_t blockSizeOverride, const std::string &perfConfig,
    InitParamsNonXDL &validParams, DerivedParams &gemmADerivedParam,
    DerivedParams &gemmBDerivedParam,
    DerivedBlockGemmParams &blockGemmDerivedParam,
    DerivedOutParams &gemmCDerivedParam, uint32_t &gridSize) {

  ConvolutionContext ctx = populateConvContext(op);

  GemmSize gemmSize;
  obtainGemmSize(ctx, gemmSize);

  if (!perfConfig.empty()) {
    // Under two scenarios can we receive a perfConfig:
    // 1. This is tuning mode
    // 2. This is running mode and we have succeeded with a perfdb load
    bool isValidPerfConfig = validParams.deserialize(perfConfig);
    if (isValidPerfConfig) {
      LLVM_DEBUG(llvm::dbgs() << genDebugForParams(validParams));
      return populateDerived(ctx, validParams, gemmSize, gemmADerivedParam,
                             gemmBDerivedParam, blockGemmDerivedParam,
                             gemmCDerivedParam, gridSize);
    }
    // Signal the client if perfCofnig is passed in but is invalid
    return failure();
  }

#if __MLIR_ENABLE_SQLITE__
  std::string solverId;
  if (ctx.opType == ConvOpType::Fwd) {
    solverId = "ConvHipImplicitGemmV4R4Fwd";
  } else if (ctx.opType == ConvOpType::BwdData) {
    solverId = "ConvHipImplicitGemmBwdDataV1R1";
  } else {
    solverId = "ConvHipImplicitGemmV4R4WrW";
  }

  SQLitePerfDb perfDb = getDb(ctx.arch, ctx.num_cu);
  bool loadRes = perfDb.load(ctx, solverId, validParams);
  if (loadRes) {
    LLVM_DEBUG(llvm::dbgs() << genDebugForParams(validParams));
    return populateDerived(ctx, validParams, gemmSize, gemmADerivedParam,
                           gemmBDerivedParam, blockGemmDerivedParam,
                           gemmCDstPerWrite, gridSize);
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "DB load failed, falling back to backup path.\n");
  }
#endif // MLIR_ENABLE_SQLITE

  // Backup path: Use the set of default tuning parameters
  LogicalResult res = failure();

  for (auto &params : initParameters) {
    // We have an override on the blockSize, only loop through the
    // initParameters with the same blockSize
    if ((blockSizeOverride != 0) && (blockSizeOverride != params.blockSize)) {
      continue;
    }

    res = populateDerived(ctx, params, gemmSize, gemmADerivedParam,
                          gemmBDerivedParam, blockGemmDerivedParam,
                          gemmCDerivedParam, gridSize);
    if (failed(res)) {
      continue;
    }

    validParams = params;
    break;
  }

  if (failed(res)) {
    // All initParameters have failed, shouldn't happen
    LLVM_DEBUG(llvm::dbgs() << "FATAL ERROR! COULD NOT FIND VALID TUNING"
                            << " PARAMETERS!\n");

      LLVM_DEBUG(llvm::dbgs() << "BUT PADDING KERNEL CAN EXECUTE IT\n");

      for (auto &params : initParameters) {
        res = populatePaddingKernelDerived(
            ctx, params, gemmSize, gemmADerivedParam, gemmBDerivedParam,
            blockGemmDerivedParam, gemmCDerivedParam, gridSize);

        if (failed(res)) {
          continue;
        }

        validParams = params;
        break;
      }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Successfully picked tuning params from backup"
                            << " path.\n");
  }

  return res;
}

ArrayRef<InitParamsNonXDL>
PopulateParams::getTuningParameters(ConvOpType dir, Type dataType) const {
  return {initParameters, nInitParameters};
}

const InitParams &PopulateParams::getUniversalParameters() const {
  return universalParameters;
}

LogicalResult PopulateParams::isValidGemm(const InitParamsNonXDL &param,
                                          GemmSize &gemmSize) {
  if (!(gemmSize.gemmM % param.gemmMPerBlock == 0 &&
        gemmSize.gemmN % param.gemmNPerBlock == 0 &&
        gemmSize.gemmK % param.gemmKPerBlock == 0)) {
    return failure();
  }
  return success();
}

/// Xdlops
// clang-format off
const InitParamsXDL
PopulateParamsXDL::initParameters[PopulateParamsXDL::nInitParameters] = {
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

const InitParamsXDL
PopulateParamsXDL::initParametersForwardI8[
  PopulateParamsXDL::nInitParametersForwardI8] = {
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

const InitParams PopulateParamsXDL::universalParameters = {32, 64, 4};

uint32_t PopulateParamsXDL::obtainBlockSize(const InitParamsXDL &params,
                                            int64_t waveSize) {
  return waveSize * params.gemmNPerBlock * params.gemmMPerBlock /
         (params.gemmMPerWave * params.gemmNPerWave);
}

LogicalResult PopulateParamsXDL::getKBlocks(ConvolutionContext &ctx,
                                            const InitParamsXDL &params,
                                            int64_t &gemmKBlocks) {
  ConvolutionDims convDims = ctx.getConvDims();

  return calculateKBlockNum(convDims, params.gemmMPerBlock,
                            params.gemmNPerBlock, params.gemmKPerBlock,
                            params.gemmKPack, ctx.num_cu, gemmKBlocks);
}

LogicalResult PopulateParamsXDL::calculateGemmABlockCopyPerformanceParameters(
    const InitParamsXDL &param, ConvolutionContext &ctx,
    DerivedParams &derived) {
  int64_t blockSize = obtainBlockSize(param, waveSize);
  return calculateInputDerivedParams(param, blockSize, ctx, true, derived);
}

LogicalResult PopulateParamsXDL::calculateGemmBBlockCopyPerformanceParameters(
    const InitParamsXDL &param, ConvolutionContext &ctx,
    DerivedParams &derived) {
  int64_t blockSize = obtainBlockSize(param, waveSize);
  return calculateInputDerivedParams(param, blockSize, ctx, false, derived);
}

LogicalResult PopulateParamsXDL::calculateLdsNumberOfByte(
    const InitParamsXDL &param, const ConvolutionContext &ctx,
    DerivedParams gemmADerived, DerivedParams gemmBDerived, size_t &ldsSize) {

  int64_t threadGemmDataPerRead_GemmM =
      param.gemmMPerBlock / gemmADerived.clusterLenGemmPos2;
  int64_t threadGemmDataPerRead_GemmN =
      param.gemmNPerBlock / gemmBDerived.clusterLenGemmPos2;

  const auto max_lds_align =
      math_util::lcm(threadGemmDataPerRead_GemmM, threadGemmDataPerRead_GemmN);

  const auto a_block_space =
      param.gemmKPerBlock *
      math_util::integer_least_multiple(param.gemmMPerBlock, max_lds_align);
  const auto b_block_space =
      param.gemmKPerBlock *
      math_util::integer_least_multiple(param.gemmNPerBlock, max_lds_align);

  ldsSize = (a_block_space + b_block_space) * sizeof(float);

  if (ldsSize > 64 * 1024) {
    return failure();
  }

  return success();
}

LogicalResult PopulateParamsXDL::isValidBlockwiseGemmXDLOPS(
    const InitParamsXDL &param, ConvolutionContext &ctx, uint32_t blockSize) {
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

  if (!std::any_of(validWaveGemmSize.cbegin(), validWaveGemmSize.cend(),
                   [param](const auto it) noexcept -> bool {
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
    LLVM_DEBUG(llvm::dbgs() << "Invalid KPACK tuning parameter: "
                            << param.gemmKPack << "\n");
    return failure();
  }
  if ((dataType.isF16() || dataType.isBF16()) && (param.gemmKPack != 1) &&
      ((param.gemmKPack < 4) || (param.gemmKPack > 8))) {
    LLVM_DEBUG(llvm::dbgs() << "Invalid KPACK tuning parameter: "
                            << param.gemmKPack << "\n");
    return failure();
  }

  // XXX FIXME: Temporarily reject KPACK=4 for fp32 backward weight
  // convolution. It has been verified some configs would cause intermittent
  // failures.
  // TODO(whchung): Get to the bottom of this.
  if ((param.gemmKPack == 4) && (ctx.getOpType() == ConvOpType::BwdWeight) &&
      dataType.isF32()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Invalid config: fp32 XDLOPS backward weight convolution "
                  "with KPACK=4\n");
    return failure();
  }

  return success();
}

LogicalResult PopulateParamsXDL::populateDerived(
    ConvolutionContext &ctx, const InitParamsXDL &params, GemmSize &gemmSize,
    DerivedParams &gemmADerivedParam, DerivedParams &gemmBDerivedParam,
    DerivedOutParams &gemmCDerivedParam, uint32_t &blockSize,
    uint32_t &gridSize, int64_t &gemmKBlocks) {
  LogicalResult res = isValidGemm(params, gemmSize);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Gemm sizes, M: " << gemmSize.gemmM
               << " N: " << gemmSize.gemmN << " K: " << gemmSize.gemmK << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Gemm size and gemm/block "
                            << "size does not divide exactly.\n");
    return failure();
  }

  if (ctx.opType == ConvOpType::BwdData &&
      failed(isValidGridGemmXdlops(gemmSize))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Invalid XDLops gemm sizes for backward data.\n");
    return failure();
  }
  blockSize = obtainBlockSize(params, waveSize);

  res = isValidBlockwiseGemmXDLOPS(params, ctx, blockSize);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Invalid XDLOPS gemm.\n");
    return failure();
  }

  res = calculateGemmABlockCopyPerformanceParameters(params, ctx,
                                                     gemmADerivedParam);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmA tuning parameter "
                            << " size.\n");
    return failure();
  }

  res = calculateGemmBBlockCopyPerformanceParameters(params, ctx,
                                                     gemmBDerivedParam);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmB tuning parameter "
                            << " size.\n");
    return failure();
  }

  std::size_t ldsSize = 0;
  res = calculateLdsNumberOfByte(params, ctx, gemmADerivedParam,
                                 gemmBDerivedParam, ldsSize);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "LDS size too large.\n");
    return failure();
  }

  // parameters derivable from tunable parameters.
  gemmKBlocks = 1;
  if (ctx.opType == ConvOpType::BwdWeight &&
      (ctx.getDataType().isF32() || ctx.getDataType().isF16())) {
    res = getKBlocks(ctx, params, gemmKBlocks);
    if (failed(res)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Invalid tuning parameters for computing KBlocks.\n");
      return failure();
    }
  }
  gridSize = obtainGridSize(gemmSize, params) * gemmKBlocks;

  res = calculateOutputDerivedParams(params, blockSize, ctx, gemmCDerivedParam);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmC tuning parameters\n");
    return failure();
  }
  return success();
}

LogicalResult PopulateParamsXDL::populatePaddingKernelDerived(
    ConvolutionContext &ctx, const InitParamsXDL &param, GemmSize &gemmSize,
    DerivedParams &gemmADerivedParam, DerivedParams &gemmBDerivedParam,
    DerivedOutParams &gemmCDerivedParam, uint32_t &blockSize,
    uint32_t &gridSize) {

  LogicalResult res = failure();
  InitParams paddingParam = getUniversalParameters();

  // find complete config for paddingParam
  // if mblock,nblock,kblock is the same with this tuning parameter
  if (paddingParam.gemmMPerBlock != param.gemmMPerBlock ||
      paddingParam.gemmNPerBlock != param.gemmNPerBlock ||
      paddingParam.gemmKPerBlock != param.gemmKPerBlock)
    return failure();

  if (gemmSize.gemmM % param.gemmMPerBlock != 0)
    gemmSize.gemmM = gemmSize.gemmM + (param.gemmMPerBlock -
                                       gemmSize.gemmM % param.gemmMPerBlock);

  if (gemmSize.gemmN % param.gemmNPerBlock != 0)
    gemmSize.gemmN = gemmSize.gemmN + (param.gemmNPerBlock -
                                       gemmSize.gemmN % param.gemmNPerBlock);

  if (gemmSize.gemmK % param.gemmKPerBlock != 0)
    gemmSize.gemmK = gemmSize.gemmK + (param.gemmKPerBlock -
                                       gemmSize.gemmK % param.gemmKPerBlock);

  blockSize = obtainBlockSize(param, waveSize);
  res = calculateGemmABlockCopyPerformanceParameters(param, ctx,
                                                     gemmADerivedParam);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmA tuning parameter "
                            << " size.\n");
    return failure();
  }

  res = calculateGemmBBlockCopyPerformanceParameters(param, ctx,
                                                     gemmBDerivedParam);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmB tuning parameter "
                            << " size.\n");
    return failure();
  }

  std::size_t ldsSize = 0;
  res = calculateLdsNumberOfByte(param, ctx, gemmADerivedParam,
                                 gemmBDerivedParam, ldsSize);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "LDS size too large.\n");
    return failure();
  }

  gridSize = obtainGridSize(gemmSize, param);
  res = calculateOutputDerivedParams(param, blockSize, ctx, gemmCDerivedParam);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmC tuning parameters\n");
    return failure();
  }
  return success();
}

LogicalResult PopulateParamsXDL::isValidGridGemmXdlops(GemmSize &gemmSize) {
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

LogicalResult PopulateParamsXDL::obtainTuningParameters(
    Operation *op, uint32_t blockSizeOverride, const std::string &perfConfig,
    InitParamsXDL &validParams, DerivedParams &gemmADerivedParam,
    DerivedParams &gemmBDerivedParam, DerivedOutParams &gemmCDerivedParam,
    uint32_t &blockSize, uint32_t &gridSize, int64_t &gemmKBlocks) {

  ConvolutionContext ctx = populateConvContext(op);

  GemmSize gemmSize;
  obtainGemmSize(ctx, gemmSize);

  if (!perfConfig.empty()) {
    // Under two scenarios can we receive a perfConfig:
    // 1. This is tuning mode
    // 2. This is running mode and we have succeeded with a perfdb load
    bool isValidPerfConfig = validParams.deserialize(perfConfig);
    if (isValidPerfConfig) {
      LLVM_DEBUG(llvm::dbgs() << genDebugForParams(validParams));
      return populateDerived(ctx, validParams, gemmSize, gemmADerivedParam,
                             gemmBDerivedParam, gemmCDerivedParam, blockSize,
                             gridSize, gemmKBlocks);
    }
    // Signal the client if perfCofnig is passed in but is invalid
    return failure();
  }

#if __MLIR_ENABLE_SQLITE__
  std::string solverId;
  if (ctx.opType == ConvOpType::Fwd) {
    solverId = "ConvHipImplicitGemmForwardV4R4Xdlops";
  } else if (ctx.opType == ConvOpType::BwdData) {
    solverId = "ConvHipImplicitGemmBwdDataV4R1Xdlops";
  } else {
    solverId = "ConvHipImplicitGemmWrwV4R4Xdlops";
  }

  SQLitePerfDb perfDb = getDb(ctx.arch, ctx.num_cu);
  bool loadRes = perfDb.load(ctx, solverId, validParams);
  if (loadRes) {
    LLVM_DEBUG(llvm::dbgs() << genDebugForParams(validParams));
    return populateDerived(ctx, validParams, gemmSize, gemmADerivedParam,
                           gemmBDerivedParam, blockSize, gridSize);
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "DB load failed, falling back to backup path.\n");
  }
#endif // MLIR_ENABLE_SQLITE

  LogicalResult res = failure();
  for (auto &params : getTuningParameters(ctx.getOpType(), ctx.getDataType())) {
    blockSize = obtainBlockSize(params, waveSize);
    // We have an override on the blockSize, only loop through the
    // initParameters with the same blockSize
    if ((blockSizeOverride != 0) && (blockSizeOverride != blockSize)) {
      continue;
    }

    res = populateDerived(ctx, params, gemmSize, gemmADerivedParam,
                          gemmBDerivedParam, gemmCDerivedParam, blockSize,
                          gridSize, gemmKBlocks);
    if (failed(res)) {
      continue;
    }

    validParams = params;
    break;
  }

  if (failed(res)) {
    // All initParameters have failed, shouldn't happen
    LLVM_DEBUG(llvm::dbgs() << "FATAL ERROR! COULD NOT FIND VALID TUNING"
                            << " PARAMETERS!\n");

      LLVM_DEBUG(llvm::dbgs() << "BUT PADDING KERNEL CAN EXECUTE IT\n");
      for (auto &params :
           getTuningParameters(ctx.getOpType(), ctx.getDataType())) {
        res = populatePaddingKernelDerived(
            ctx, params, gemmSize, gemmADerivedParam, gemmBDerivedParam,
            gemmCDerivedParam, blockSize, gridSize);

        if (failed(res)) {
          continue;
        }
        validParams = params;
        break;
      }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Successfully picked tuning params from backup"
                            << " path.\n");
  }
  LLVM_DEBUG(llvm::dbgs() << genDebugForParams(validParams) << "\n");

  return res;
}

ArrayRef<InitParamsXDL>
PopulateParamsXDL::getTuningParameters(ConvOpType dir, Type dataType) const {
  if (dataType.isInteger(8)) {
    return {initParametersForwardI8, nInitParametersForwardI8};
  }

  return {initParameters, nInitParameters};
}

const InitParams &PopulateParamsXDL::getUniversalParameters() const {
  return universalParameters;
}

LogicalResult PopulateParamsXDL::isValidGemm(const InitParamsXDL &param,
                                             GemmSize &gemmSize) const {
  if (!(gemmSize.gemmM % param.gemmMPerBlock == 0 &&
        gemmSize.gemmN % param.gemmNPerBlock == 0 &&
        gemmSize.gemmK % (param.gemmKPerBlock * param.gemmKPack) == 0)) {
    return failure();
  }
  return success();
}

Optional<GemmContext> mlir::rock::requiredPadding(Attribute params,
                                                    GemmContext gemmSize) {
  int64_t kPerBlock, mPerBlock, nPerBlock;
  if (auto generalParams = params.dyn_cast<GeneralGemmParamsAttr>()) {
    kPerBlock = generalParams.getKPerBlock();
    mPerBlock = generalParams.getMPerBlock();
    nPerBlock = generalParams.getNPerBlock();
  } else if (auto xdlopsParams = params.dyn_cast<XdlopsGemmParamsAttr>()) {
    kPerBlock = xdlopsParams.getKPerBlock();
    mPerBlock = xdlopsParams.getMPerBlock();
    nPerBlock = xdlopsParams.getNPerBlock();
  } else {
    llvm_unreachable("The tuning paramaters are general or xdlops");
  }

  int64_t kExtra = kPerBlock - math_util::mod_1_to_n(gemmSize.k, kPerBlock);
  int64_t mExtra = mPerBlock - math_util::mod_1_to_n(gemmSize.m, mPerBlock);
  int64_t nExtra = nPerBlock - math_util::mod_1_to_n(gemmSize.n, nPerBlock);
  if (mExtra == 0 && kExtra == 0 && nExtra == 0)
    return None;
  return GemmContext(mExtra, kExtra, nExtra);
}
