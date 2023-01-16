#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/IR/ConvolutionDims.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/MfmaInsnGroup.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/Tuning/SqliteDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
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
  return os;
}

static int64_t obtainGridSize(const GemmSize &gemmSize,
                              const InitParams &param) {
  return (gemmSize.m / param.gemmMPerBlock) *
         (gemmSize.n / param.gemmNPerBlock) * gemmSize.g;
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

LogicalResult PopulateParams::calculateBlockGemmPerformanceParameters(
    const InitParamsNonXDL &param, RockGemmWrapperInterface op) {

  FailureOr<GeneralGemmBlockStructure> maybeDerived =
      deriveGeneralGemmBlockStructure(param.blockSize);
  if (failed(maybeDerived))
    return failure();
  GeneralGemmBlockStructure derived = std::move(*maybeDerived);

  if (!(param.gemmMPerThread >= 2 && param.gemmMPerThread <= 4))
    return failure();

  if (!(param.gemmNPerThread >= 2 && param.gemmNPerThread <= 4))
    return failure();

  if (!(param.gemmMPerBlock % param.gemmMPerThread == 0 &&
        param.gemmNPerBlock % param.gemmNPerThread == 0))
    return failure();

  int64_t threadGemmMPerCluster = param.gemmMPerThread *
                                  derived.mThreadsPerCuwave *
                                  derived.mCuwavesPerBlock;
  int64_t threadGemmNPerCluster = param.gemmNPerThread *
                                  derived.nThreadsPerCuwave *
                                  derived.nCuwavesPerBlock;

  if ((param.gemmMPerBlock % threadGemmMPerCluster != 0) ||
      (param.gemmNPerBlock % threadGemmNPerCluster != 0)) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "M per block or N per block aren't divisible by M/N per cluster\n");
    return failure();
  }

  return success();
}

LogicalResult PopulateParams::populateDerived(RockGemmWrapperInterface op,
                                              const InitParamsNonXDL &params,
                                              GemmSize &gemmSize,
                                              uint32_t &gridSize) {
  auto gemmExtraPad =
      calculatePadding(params.gemmKPerBlock, params.gemmMPerBlock,
                       params.gemmNPerBlock, gemmSize);
  if (gemmExtraPad.has_value()) {
    gemmSize.m += gemmExtraPad->m;
    gemmSize.k += gemmExtraPad->k;
    gemmSize.n += gemmExtraPad->n;
  }

  LogicalResult res = calculateBlockGemmPerformanceParameters(params, op);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent blockGemm tuning parameter "
                            << " size.\n");
    return failure();
  }

  gridSize = obtainGridSize(gemmSize, params);
  return success();
}

LogicalResult PopulateParams::obtainTuningParameters(
    RockGemmWrapperInterface op, uint32_t blockSizeOverride,
    const std::string &perfConfig, InitParamsNonXDL &validParams,
    uint32_t &gridSize) {

  GemmSize gemmSize = op.getGemmSize();

  if (!perfConfig.empty()) {
    // Under two scenarios can we receive a perfConfig:
    // 1. This is tuning mode
    // 2. This is running mode and we have succeeded with a perfdb load
    bool isValidPerfConfig = validParams.deserialize(perfConfig);
    if (isValidPerfConfig) {
      LLVM_DEBUG(llvm::dbgs() << genDebugForParams(validParams));
      return populateDerived(op, validParams, gemmSize, gridSize);
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
    return populateDerived(ctx, validParams, gemmSize, gridSize);
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "DB load failed, falling back to backup path.\n");
  }
#endif // MLIR_ENABLE_SQLITE

  // Backup path: Use the set of default tuning parameters
  LogicalResult res = failure();
  std::vector<InitParamsNonXDL> paramSets =
      getTuningParameters(op.getKernelType(), op.getInputType());
  for (auto &params : orderInitParams(paramSets, gemmSize)) {
    // We have an override on the blockSize, only loop through the
    // initParameters with the same blockSize
    if ((blockSizeOverride != 0) && (blockSizeOverride != params.blockSize)) {
      continue;
    }

    res = populateDerived(op, params, gemmSize, gridSize);
    if (failed(res)) {
      continue;
    }

    validParams = params;
    break;
  }

  return res;
}

std::vector<InitParamsNonXDL>
PopulateParams::getTuningParameters(KernelType opType, Type dataType) const {
  ArrayRef<InitParamsNonXDL> params = {initParameters, nInitParameters};
  return std::vector<InitParamsNonXDL>(params);
}

const InitParams &PopulateParams::getUniversalParameters() const {
  return universalParameters;
}

LogicalResult PopulateParams::isValidGemm(const InitParamsNonXDL &param,
                                          const GemmSize &gemmSize) const {
  if (!(gemmSize.m % param.gemmMPerBlock == 0 &&
        gemmSize.n % param.gemmNPerBlock == 0 &&
        gemmSize.k % param.gemmKPerBlock == 0)) {
    return failure();
  }
  return success();
}

static int64_t calculatePaddingComplexity(const GemmSize &paddingAmount,
                                          const GemmSize &gemmSize) {
  int64_t nonPaddedComplexity = gemmSize.m * gemmSize.k * gemmSize.n;
  int64_t paddedComplexity = (gemmSize.m + paddingAmount.m) *
                             (gemmSize.k + paddingAmount.k) *
                             (gemmSize.n + paddingAmount.n);
  return paddedComplexity - nonPaddedComplexity;
}

int64_t PopulateParams::calculatePaddingAmount(const InitParamsNonXDL &params,
                                               const GemmSize &gemmSize) const {
  Optional<GemmSize> maybeGemmExtraPad =
      calculatePadding(params.gemmKPerBlock, params.gemmMPerBlock,
                       params.gemmNPerBlock, gemmSize);
  if (maybeGemmExtraPad.has_value()) {
    return calculatePaddingComplexity(maybeGemmExtraPad.value(), gemmSize);
  }
  return 0;
}

int64_t
PopulateParamsXDL::calculatePaddingAmount(const InitParamsXDL &params,
                                          const GemmSize &gemmSize) const {
  Optional<GemmSize> maybeGemmExtraPad =
      calculatePadding(params.gemmKPerBlock, params.gemmMPerBlock,
                       params.gemmNPerBlock, gemmSize, params.gemmKPack);
  if (maybeGemmExtraPad.has_value()) {
    return calculatePaddingComplexity(maybeGemmExtraPad.value(), gemmSize);
  }
  return 0;
}

/// Xdlops
// clang-format off
const InitParamsXDL
PopulateParamsXDL::initParameters[PopulateParamsXDL::nInitParameters] = {
  // M/block N/block K/block M/wave N/wave kPack forceUnroll bCopyMore
  {128, 128, 4, 64, 64, 4, true, true},
  {32, 64, 4, 32, 64, 4, true, true},

  {128, 128, 8, 64, 64, 1, true, true},
  {128, 128, 16, 64, 64, 1, true, true},
  {8, 64, 8, 8, 64, 1, true, true},
  {4, 64, 16, 4, 64, 1, true, true},
  {32, 64, 4, 32, 64, 1, true, true},
  {16, 16, 16, 16, 16, 1, true, true},
  {16, 16, 4, 16, 16, 1, true, true},
};

const InitParamsXDL
PopulateParamsXDL::initParametersForwardI8[
  PopulateParamsXDL::nInitParametersForwardI8] = {
  // M/block N/block K/block M/wave N/wave kPack forceUnroll bCopyMore
  // kpack for int8 must be larger than kbase, which means
  // kpack must be at least 4, once enabled.
  {64, 64, 8, 32, 32, 8, true, true},
  {64, 64, 8, 32, 32, 4, true, true},
  {32, 32, 8, 16, 16, 8, true, true},
  {32, 32, 8, 16, 16, 4, true, true},
  // The 32 x 32 xdlops k/block must be at least 8
  {64, 64, 16, 32, 32, 1, true, true},
  {64, 64, 8, 32, 32, 1, true, true},
  {32, 32, 16, 32, 32, 1, true, true},
  {32, 32, 8, 32, 32, 1, true, true},
  // The 16 x 16 xdlops k/block must be at least 16
  {32, 32, 32, 16, 16, 1, true, true},
  {32, 32, 16, 16, 16, 1, true, true},
  {16, 16, 32, 16, 16, 1, true, true},
  {16, 16, 16, 16, 16, 1, true, true},
};
// clang-format on

const InitParams PopulateParamsXDL::universalParameters = {32, 64, 4};

uint32_t PopulateParamsXDL::obtainBlockSize(const InitParamsXDL &params,
                                            int64_t waveSize) {
  return waveSize * params.gemmNPerBlock * params.gemmMPerBlock /
         (params.gemmMPerWave * params.gemmNPerWave);
}

LogicalResult PopulateParamsXDL::getKBlocks(Conv2DBwdWeightOp op,
                                            const GemmSize &gemmSize,
                                            const InitParamsXDL &params,
                                            int64_t &gemmKBlocks) {
  auto convDims = ConvolutionDims::fromOp(op);

  return calculateKBlockNum(convDims, gemmSize, params.gemmMPerBlock,
                            params.gemmNPerBlock, params.gemmKPerBlock,
                            params.gemmKPack, op.getNumCu(), gemmKBlocks);
}

LogicalResult
PopulateParamsXDL::isValidBlockwiseGemmXDLOPS(const InitParamsXDL &param,
                                              RockGemmWrapperInterface op,
                                              uint32_t blockSize) {
  // TBD: support fp16/bf16

  Type dataType = op.getInputType();
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
      std::make_tuple(128, 128, 1),
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

  // Sledgehammer hotfix because not unrolling sometimes makes the register
  // allocator break. This should be refined quickly.
  if (0&&param.gemmAThreadCopyMoreGemmK == false) {
    return failure();
  }

  // Reject invalid KPACK values.
  auto maybeMfmaInsnGroup =
      MfmaInsnGroup::select(dataType, param.gemmMPerWave, param.gemmNPerWave);
  if (failed(maybeMfmaInsnGroup)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to select xdlops instruction group.\n");
    return failure();
  }
  MfmaInsnGroup mfmaGroup = *maybeMfmaInsnGroup;
  if (!mfmaGroup.isCoherentWithK(param.gemmKPack, param.gemmKPerBlock)) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Mfma instruction group selection is not compatible with k.\n");
    return failure();
  }

  return success();
}

LogicalResult PopulateParamsXDL::populateDerived(RockGemmWrapperInterface op,
                                                 const InitParamsXDL &params,
                                                 GemmSize &gemmSize,
                                                 uint32_t &blockSize,
                                                 uint32_t &gridSize,
                                                 int64_t &gemmKBlocks) {
  bool requiredPadding = false;
  auto gemmExtraPad =
      calculatePadding(params.gemmKPerBlock, params.gemmMPerBlock,
                       params.gemmNPerBlock, gemmSize, params.gemmKPack);
  if (gemmExtraPad.has_value()) {
    gemmSize.m += gemmExtraPad->m;
    gemmSize.k += gemmExtraPad->k;
    gemmSize.n += gemmExtraPad->n;
    requiredPadding = true;
  }

  blockSize = obtainBlockSize(params, waveSize);

  LogicalResult res = isValidBlockwiseGemmXDLOPS(params, op, blockSize);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Invalid XDLOPS gemm.\n");
    return failure();
  }

  gridSize = obtainGridSize(gemmSize, params);

  // parameters derivable from tunable parameters.
  gemmKBlocks = 1;
  Type inputType = op.getInputType();
  auto maybeWrwOp = dyn_cast<Conv2DBwdWeightOp>(*op);
  if (maybeWrwOp &&
      isWrWAtomicKernel(op.getGemmFeatures(), inputType, requiredPadding)) {
    res = getKBlocks(maybeWrwOp, gemmSize, params, gemmKBlocks);
    if (failed(res)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Invalid tuning parameters for computing KBlocks.\n");
      return failure();
    }
    gridSize *= gemmKBlocks;
  }

  return success();
}

LogicalResult PopulateParamsXDL::obtainTuningParameters(
    RockGemmWrapperInterface op, uint32_t blockSizeOverride,
    const std::string &perfConfig, InitParamsXDL &validParams,
    uint32_t &blockSize, uint32_t &gridSize, int64_t &gemmKBlocks) {
  GemmSize gemmSize = op.getGemmSize();

  if (!perfConfig.empty()) {
    // Under two scenarios can we receive a perfConfig:
    // 1. This is tuning mode
    // 2. This is running mode and we have succeeded with a perfdb load
    bool isValidPerfConfig = validParams.deserialize(perfConfig);
    if (isValidPerfConfig) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Got perf config: " << genDebugForParams(validParams));
      return populateDerived(op, validParams, gemmSize, blockSize, gridSize,
                             gemmKBlocks);
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
    return populateDerived(ctx, validParams, gemmSize, blockSize, gridSize);
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "DB load failed, falling back to backup path.\n");
  }
#endif // MLIR_ENABLE_SQLITE

  LogicalResult res = failure();
  std::vector<InitParamsXDL> paramSets =
      getTuningParameters(op.getKernelType(), op.getInputType());
  for (const auto &params : orderInitParams(paramSets, gemmSize)) {
    blockSize = obtainBlockSize(params, waveSize);
    // We have an override on the blockSize, only loop through the
    // initParameters with the same blockSize
    if ((blockSizeOverride != 0) && (blockSizeOverride != blockSize)) {
      continue;
    }

    res =
        populateDerived(op, params, gemmSize, blockSize, gridSize, gemmKBlocks);
    if (failed(res)) {
      continue;
    }
    validParams = params;
    break;
  }
  LLVM_DEBUG(llvm::dbgs() << genDebugForParams(validParams) << "\n");
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Couldn't pick heuristic values for ");
    LLVM_DEBUG(op->print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
  return res;
}

std::vector<InitParamsXDL>
PopulateParamsXDL::getTuningParameters(KernelType opType, Type dataType) const {
  ArrayRef<InitParamsXDL> params;
  if (dataType.isInteger(8)) {
    params = {initParametersForwardI8, nInitParametersForwardI8};
  } else {
    params = {initParameters, nInitParameters};
  }
  std::vector<InitParamsXDL> res;
  // Only return valid XDLOp params
  std::copy_if(
      params.begin(), params.end(), std::back_inserter(res),
      [&](const InitParamsXDL &param) {
        auto maybeMfmaInsnGroup = MfmaInsnGroup::select(
            dataType, param.gemmMPerWave, param.gemmNPerWave);
        if (failed(maybeMfmaInsnGroup)) {
          return false;
        }
        MfmaInsnGroup mfmaGroup = *maybeMfmaInsnGroup;
        if (!mfmaGroup.isCoherentWithK(param.gemmKPack, param.gemmKPerBlock)) {
          return false;
        }
        return true;
      });
  return res;
}

const InitParams &PopulateParamsXDL::getUniversalParameters() const {
  return universalParameters;
}

LogicalResult PopulateParamsXDL::isValidGemm(const InitParamsXDL &param,
                                             const GemmSize &gemmSize) const {
  if (!(gemmSize.m % param.gemmMPerBlock == 0 &&
        gemmSize.n % param.gemmNPerBlock == 0 &&
        gemmSize.k % (param.gemmKPerBlock * param.gemmKPack) == 0)) {
    return failure();
  }
  return success();
}

Optional<GemmSize> mlir::rock::calculatePadding(int64_t kPerBlock,
                                                int64_t mPerBlock,
                                                int64_t nPerBlock,
                                                const GemmSize &gemmSize,
                                                int64_t kPack) {
  int64_t kExtra = (kPerBlock * kPack) -
                   math_util::mod_1_to_n(gemmSize.k, kPerBlock * kPack);
  int64_t mExtra = mPerBlock - math_util::mod_1_to_n(gemmSize.m, mPerBlock);
  int64_t nExtra = nPerBlock - math_util::mod_1_to_n(gemmSize.n, nPerBlock);
  if (mExtra == 0 && kExtra == 0 && nExtra == 0)
    return None;
  return GemmSize(0, mExtra, kExtra, nExtra);
}

Optional<GemmSize> mlir::rock::requiredPadding(Attribute params,
                                               GemmSize gemmSize) {
  int64_t kPerBlock, mPerBlock, nPerBlock;
  int64_t kPack = 1;
  if (auto generalParams = params.dyn_cast<GeneralGemmParamsAttr>()) {
    kPerBlock = generalParams.getKPerBlock();
    mPerBlock = generalParams.getMPerBlock();
    nPerBlock = generalParams.getNPerBlock();
  } else if (auto xdlopsParams = params.dyn_cast<XdlopsGemmParamsAttr>()) {
    kPerBlock = xdlopsParams.getKPerBlock();
    mPerBlock = xdlopsParams.getMPerBlock();
    nPerBlock = xdlopsParams.getNPerBlock();
    kPack = xdlopsParams.getKpack();
  } else {
    llvm_unreachable("The tuning paramaters are general or xdlops");
  }
  return calculatePadding(kPerBlock, mPerBlock, nPerBlock, gemmSize, kPack);
}
