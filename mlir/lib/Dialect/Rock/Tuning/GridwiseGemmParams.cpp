#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/IR/ConvolutionDims.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/MfmaInsnGroup.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/WmmaInsnGroup.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

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
const InitParamsNonAccel
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

PopulateParamsInfo PopulateParamsInfo::fromOp(RockGemmWrapperInterface op) {
  PopulateParamsInfo info{op.getGemmSize(), op.getArch(),  op.getGemmFeatures(),
                          op.getAType(),    op.getBType(), op.getKernelType()};

  if (auto convOp = dyn_cast<Conv2DBwdWeightOp>(*op)) {
    auto convDims = ConvolutionDims::fromOp(op);
    info.numCu = convOp.getNumCu();
    info.batchSize = convDims.n;
  }
  return info;
}

std::optional<GemmSize> mlir::rock::calculatePadding(int64_t kPerBlock,
                                                     int64_t mPerBlock,
                                                     int64_t nPerBlock,
                                                     const GemmSize &gemmSize,
                                                     int64_t kPack) {
  int64_t kExtra = (kPerBlock * kPack) -
                   math_util::mod_1_to_n(gemmSize.k, kPerBlock * kPack);
  int64_t mExtra = mPerBlock - math_util::mod_1_to_n(gemmSize.m, mPerBlock);
  int64_t nExtra = nPerBlock - math_util::mod_1_to_n(gemmSize.n, nPerBlock);
  if (mExtra == 0 && kExtra == 0 && nExtra == 0)
    return std::nullopt;
  return GemmSize(0, mExtra, kExtra, nExtra);
}

std::optional<GemmSize> mlir::rock::requiredPadding(Attribute params,
                                                    GemmSize gemmSize) {
  int64_t kPerBlock, mPerBlock, nPerBlock;
  int64_t kPack = 1;
  if (auto generalParams = params.dyn_cast<GeneralGemmParamsAttr>()) {
    kPerBlock = generalParams.getKPerBlock();
    mPerBlock = generalParams.getMPerBlock();
    nPerBlock = generalParams.getNPerBlock();
  } else if (auto accelParams =
                 params.dyn_cast<RockAccelTuningParamAttrInterface>()) {
    kPerBlock = accelParams.getKpackPerBlock();
    mPerBlock = accelParams.getMPerBlock();
    nPerBlock = accelParams.getNPerBlock();
    kPack = accelParams.getKpack();
  } else {
    llvm_unreachable("The tuning paramaters are general or xdlops");
  }
  return calculatePadding(kPerBlock, mPerBlock, nPerBlock, gemmSize, kPack);
}

LogicalResult PopulateParams::calculateBlockGemmPerformanceParameters(
    const InitParamsNonAccel &param) {

  FailureOr<GeneralGemmBlockStructure> maybeDerived =
      deriveGeneralGemmBlockStructure(param.blockSize);
  if (failed(maybeDerived))
    return failure();
  GeneralGemmBlockStructure derived = *maybeDerived;

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

LogicalResult PopulateParams::populateDerived(const InitParamsNonAccel &params,
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

  LogicalResult res = calculateBlockGemmPerformanceParameters(params);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent blockGemm tuning parameter "
                            << " size.\n");
    return failure();
  }

  gridSize = obtainGridSize(gemmSize, params);
  return success();
}

LogicalResult PopulateParams::obtainTuningParameters(
    const PopulateParamsInfo &info, uint32_t blockSizeOverride,
    const std::string &perfConfig, InitParamsNonAccel &validParams,
    uint32_t &gridSize) {

  if (!perfConfig.empty()) {
    // Under two scenarios can we receive a perfConfig:
    // 1. This is tuning mode
    // 2. This is running mode and we have succeeded with a perfdb load
    bool isValidPerfConfig = validParams.deserialize(perfConfig);
    if (isValidPerfConfig) {
      LLVM_DEBUG(llvm::dbgs() << genDebugForParams(validParams));
      GemmSize paddedGemmSize = info.gemmSize;
      return populateDerived(validParams, paddedGemmSize, gridSize);
    }
    // Signal the client if perfCofnig is passed in but is invalid
    return failure();
  }

  // Backup path: Use the set of default tuning parameters
  LogicalResult res = failure();
  auto paramSets =
      getTuningParameters(info.kernelType, info.gemmAType, info.gemmBType);
  for (auto &params : orderInitParams(paramSets, info.gemmSize)) {
    // We have an override on the blockSize, only loop through the
    // initParameters with the same blockSize
    if ((blockSizeOverride != 0) && (blockSizeOverride != params.blockSize)) {
      continue;
    }

    GemmSize paddedGemmSize = info.gemmSize;
    res = populateDerived(params, paddedGemmSize, gridSize);
    if (failed(res)) {
      continue;
    }

    validParams = params;
    break;
  }

  return res;
}

LogicalResult PopulateParams::obtainTuningParameters(
    RockGemmWrapperInterface op, uint32_t blockSizeOverride,
    const std::string &perfConfig, InitParamsNonAccel &validParams,
    uint32_t &gridSize) {
  PopulateParamsInfo info = PopulateParamsInfo::fromOp(op);
  return obtainTuningParameters(info, blockSizeOverride, perfConfig,
                                validParams, gridSize);
}

std::vector<InitParamsNonAccel>
PopulateParams::getTuningParameters(KernelType opType, Type dataTypeA,
                                    Type dataTypeB) const {
  ArrayRef<InitParamsNonAccel> params = {initParameters, nInitParameters};
  return std::vector<InitParamsNonAccel>(params);
}

static int64_t calculatePaddingComplexity(const GemmSize &paddingAmount,
                                          const GemmSize &gemmSize) {
  int64_t nonPaddedComplexity = gemmSize.m * gemmSize.k * gemmSize.n;
  int64_t paddedComplexity = (gemmSize.m + paddingAmount.m) *
                             (gemmSize.k + paddingAmount.k) *
                             (gemmSize.n + paddingAmount.n);
  return paddedComplexity - nonPaddedComplexity;
}

int64_t PopulateParams::calculatePaddingAmount(const InitParamsNonAccel &params,
                                               const GemmSize &gemmSize) const {
  std::optional<GemmSize> maybeGemmExtraPad =
      calculatePadding(params.gemmKPerBlock, params.gemmMPerBlock,
                       params.gemmNPerBlock, gemmSize);
  if (maybeGemmExtraPad.has_value()) {
    return calculatePaddingComplexity(maybeGemmExtraPad.value(), gemmSize);
  }
  return 0;
}

// Acceleration common interface implementation
std::unique_ptr<PopulateParamsAccel>
PopulateParamsAccel::select(GemmFeatures features) {
  if (bitEnumContainsAll(features, GemmFeatures::mfma)) {
    return std::make_unique<PopulateParamsXDL>();
  } else if (bitEnumContainsAll(features, GemmFeatures::wmma)) {
    return std::make_unique<PopulateParamsWmma>();
  } else {
    return nullptr;
  }
}

int64_t
PopulateParamsAccel::calculatePaddingAmount(const InitParamsAccel &params,
                                            const GemmSize &gemmSize) const {
  std::optional<GemmSize> maybeGemmExtraPad =
      calculatePadding(params.gemmKPerBlock, params.gemmMPerBlock,
                       params.gemmNPerBlock, gemmSize, params.gemmKPack);
  if (maybeGemmExtraPad.has_value()) {
    return calculatePaddingComplexity(maybeGemmExtraPad.value(), gemmSize);
  }
  return 0;
}

uint32_t PopulateParamsAccel::obtainBlockSize(const InitParamsAccel &params,
                                              int64_t waveSize) {
  return waveSize * params.gemmNPerBlock * params.gemmMPerBlock /
         (params.gemmMPerWave * params.gemmNPerWave);
}

LogicalResult PopulateParamsAccel::getKBlocks(const int64_t batchSize,
                                              const GemmSize &gemmSize,
                                              const InitParamsAccel &params,
                                              int64_t &gemmKBlocks,
                                              uint32_t numCu) {
  return calculateKBlockNum(batchSize, gemmSize, params.gemmMPerBlock,
                            params.gemmNPerBlock, params.gemmKPerBlock,
                            params.gemmKPack, numCu, gemmKBlocks);
}

LogicalResult
PopulateParamsAccel::populateDerived(const InitParamsAccel &params,
                                     const PopulateParamsInfo &info,
                                     GemmSize &gemmSize, uint32_t &blockSize,
                                     uint32_t &gridSize, int64_t &gemmKBlocks) {
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

  const int64_t waveSize = mlir::rock::lookupArchInfo(info.arch).waveSize;
  blockSize = obtainBlockSize(params, waveSize);

  LogicalResult res = isValidBlockwiseGemm(
      params, info.gemmAType, info.gemmBType, info.arch, blockSize);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Invalid accelerated gemm.\n");
    return failure();
  }

  gridSize = obtainGridSize(gemmSize, params);

  // parameters derivable from tunable parameters.
  gemmKBlocks = 1;
  auto maybeWrwOp = (info.kernelType == KernelType::Conv2DBwdWeight);
  // We can pick one of the two data types as we don't support backward-weight
  // fp8 currently.
  if (maybeWrwOp &&
      isWrWAtomicKernel(info.gemmFeatures, info.gemmAType, requiredPadding)) {
    res = getKBlocks(info.batchSize, gemmSize, params, gemmKBlocks, info.numCu);
    if (failed(res)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Invalid tuning parameters for computing KBlocks.\n");
      return failure();
    }
    gridSize *= gemmKBlocks;
  }
  return success();
}

LogicalResult PopulateParamsAccel::obtainTuningParameters(
    const PopulateParamsInfo &info, uint32_t blockSizeOverride,
    const std::string &perfConfig, InitParamsAccel &validParams,
    uint32_t &blockSize, uint32_t &gridSize, int64_t &gemmKBlocks) {

  if (!perfConfig.empty()) {
    // Under two scenarios can we receive a perfConfig:
    // 1. This is tuning mode
    // 2. This is running mode and we have succeeded with a perfdb load
    bool isValidPerfConfig = validParams.deserialize(perfConfig);
    if (isValidPerfConfig) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Got perf config: " << genDebugForParams(validParams));
      GemmSize paddedGemmSize = info.gemmSize;
      return populateDerived(validParams, info, paddedGemmSize, blockSize,
                             gridSize, gemmKBlocks);
    }
    // Signal the client if perfCofnig is passed in but is invalid
    return failure();
  }

  const int64_t waveSize = mlir::rock::lookupArchInfo(info.arch).waveSize;
  LogicalResult res = failure();
  auto paramSets = getTuningParameters(info.kernelType, info.gemmAType,
                                       info.gemmBType, info.arch);

  for (const auto &params : orderInitParams(paramSets, info.gemmSize)) {
    blockSize = obtainBlockSize(params, waveSize);
    // We have an override on the blockSize, only loop through the
    // initParameters with the same blockSize
    if ((blockSizeOverride != 0) && (blockSizeOverride != blockSize)) {
      continue;
    }

    GemmSize paddedGemmSize = info.gemmSize;
    res = populateDerived(params, info, paddedGemmSize, blockSize, gridSize,
                          gemmKBlocks);
    if (failed(res)) {
      continue;
    }
    validParams = params;
    break;
  }
  LLVM_DEBUG(llvm::dbgs() << genDebugForParams(validParams) << "\n");
  return res;
}

LogicalResult PopulateParamsAccel::obtainTuningParameters(
    RockGemmWrapperInterface op, uint32_t blockSizeOverride,
    const std::string &perfConfig, InitParamsAccel &validParams,
    uint32_t &blockSize, uint32_t &gridSize, int64_t &gemmKBlocks) {
  PopulateParamsInfo info = PopulateParamsInfo::fromOp(op);
  auto res =
      obtainTuningParameters(info, blockSizeOverride, perfConfig, validParams,
                             blockSize, gridSize, gemmKBlocks);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Couldn't pick heuristic values for ");
    LLVM_DEBUG(op->print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
  return res;
}

/// Xdlops acceleration
// clang-format off
const InitParamsAccel
PopulateParamsXDL::initParameters[PopulateParamsXDL::nInitParameters] = {
  // M/block N/block K/block M/wave N/wave kPack forceUnroll bCopyMore
  {128, 128, 4, 64, 64, 4, true, true},
  {64, 64, 8, 32, 32, 4, true, true},
  {32, 64, 4, 32, 64, 4, true, true},
  {32, 64, 2, 8, 64, 4, true, true},
  {4, 64, 16, 4, 64, 1, true, true}
};

const InitParamsAccel
PopulateParamsXDL::initParametersFp16[PopulateParamsXDL::nInitParametersFp16] = {
  // M/block N/block K/block M/wave N/wave kPack forceUnroll bCopyMore
  {128,  128,  4,  64,  64,  8,  true,  true},
  {32,  128,  4,  32,  32,  8,  true,  true},
  {32, 64, 4, 32, 64, 4, true, true},
};

const InitParamsAccel
PopulateParamsXDL::initParametersForward8Bit[
  PopulateParamsXDL::nInitParametersForward8Bit] = {
  {128, 128, 8, 64, 64, 8, true, true},
  {64, 64, 8, 32, 32, 8, true, true},
  {64, 64, 8, 32, 32, 4, true, true},
  {32, 32, 8, 16, 16, 8, true, true},
  {32, 32, 8, 16, 16, 4, true, true},
};
// clang-format on

LogicalResult
PopulateParamsXDL::isValidBlockwiseGemm(const InitParamsAccel &param,
                                        Type dataTypeA, Type dataTypeB,
                                        StringRef arch, uint32_t blockSize) {

  const int64_t waveSize = mlir::rock::lookupArchInfo(arch).waveSize;
  // TBD: support fp16/bf16

  // clang-format off
  std::vector<std::tuple<int, int, int>> validWaveGemmSize =
  {
    std::make_tuple(128, 128, 2),
    std::make_tuple(128, 64, 2),
    std::make_tuple(64, 128, 2),
    std::make_tuple(64, 64, 2),
    std::make_tuple(64, 32, 2),
    std::make_tuple(32, 64, 2),
    std::make_tuple(32, 32, 2),
    std::make_tuple(64, 16, 4),
    std::make_tuple(16, 64, 4),
    std::make_tuple(16, 16, 4),
  };
  // clang-format on

  // Add broadcasts for non 8-bit types.
  bool is8BitReduceOnly = dataTypeA.isInteger(8) ||
                          dataTypeA.isFloat8E4M3FNUZ() ||
                          dataTypeA.isFloat8E5M2FNUZ();
  if (!is8BitReduceOnly) {
    validWaveGemmSize.emplace_back(8, 64, 1);
    validWaveGemmSize.emplace_back(4, 64, 1);
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
  if (blockSize < waveSize || blockSize > 4 * waveSize)
    return failure();

  if ((param.gemmMPerBlock % param.gemmMPerWave) != 0)
    return failure();

  if ((param.gemmNPerBlock % param.gemmNPerWave) != 0)
    return failure();

  // Sledgehammer hotfix because not unrolling sometimes makes the register
  // allocator break. This should be refined quickly.
  if (param.gemmAThreadCopyMoreGemmK == false) {
    return failure();
  }

  // Reject invalid KPACK values.
  auto maybeMfmaInsnGroup = MfmaInsnGroup::select(
      dataTypeA, dataTypeB, arch, param.gemmMPerWave, param.gemmNPerWave);
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

std::vector<InitParamsAccel>
PopulateParamsXDL::getTuningParameters(KernelType opType, Type dataTypeA,
                                       Type dataTypeB, StringRef arch) const {
  ArrayRef<InitParamsAccel> params;
  switch (dataTypeA.getIntOrFloatBitWidth()) {
  case 8:
    params = {initParametersForward8Bit, nInitParametersForward8Bit};
    break;
  case 16:
    params = {initParametersFp16, nInitParametersFp16};
    break;
  default:
    params = {initParameters, nInitParameters};
  }
  std::vector<InitParamsAccel> res;
  // Only return valid XDLOp params
  std::copy_if(
      params.begin(), params.end(), std::back_inserter(res),
      [&](const InitParamsAccel &param) {
        auto maybeMfmaInsnGroup = MfmaInsnGroup::select(
            dataTypeA, dataTypeB, arch, param.gemmMPerWave, param.gemmNPerWave);
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

Attribute
PopulateParamsXDL::getGemmParamsAttr(OpBuilder builder,
                                     InitParamsAccel validParams) const {
  return builder.getAttr<XdlopsGemmParamsAttr>(
      validParams.gemmKPerBlock, validParams.gemmMPerBlock,
      validParams.gemmNPerBlock, validParams.gemmKPack,
      validParams.gemmMPerWave, validParams.gemmNPerWave,
      validParams.gemmAThreadCopyMoreGemmK);
}

/// Wmma acceleration
// clang-format off
const InitParamsAccel
PopulateParamsWmma::initParametersFp16[PopulateParamsWmma::nInitParametersFp16] = {
  // M/block N/block K/block M/wave N/wave kPack forceUnroll bCopyMore
  {128,  128,  4,  64,  64,  16,  true,  true},
  {32,  128,  4,  32,  32,  16,  true,  true},
  {32, 64, 4, 32, 64, 16, true, true},
};

const InitParamsAccel
PopulateParamsWmma::initParametersForward8Bit[
  PopulateParamsWmma::nInitParametersForward8Bit] = {
  {128, 128, 8, 64, 64, 16, true, true},
  {64, 64, 4, 32, 32, 16, true, true},
  {64, 64, 8, 32, 32, 16, true, true},
  {32, 32, 4, 16, 16, 16, true, true},
  {32, 32, 8, 16, 16, 16, true, true},
};
// clang-format on

LogicalResult
PopulateParamsWmma::isValidBlockwiseGemm(const InitParamsAccel &param,
                                         Type dataTypeA, Type dataTypeB,
                                         StringRef arch, uint32_t blockSize) {

  const int64_t waveSize = mlir::rock::lookupArchInfo(arch).waveSize;

  // clang-format off
  std::vector<std::tuple<int, int, int>> validWaveGemmSize =
  {
    std::make_tuple(128, 128, 2),
    std::make_tuple(128, 64, 2),
    std::make_tuple(64, 128, 2),
    std::make_tuple(64, 64, 2),
    std::make_tuple(64, 32, 2),
    std::make_tuple(32, 64, 2),
    std::make_tuple(32, 32, 2),
    std::make_tuple(32, 16, 2),
    std::make_tuple(16, 32, 2),
    std::make_tuple(32, 32, 2),
    std::make_tuple(64, 16, 2),
    std::make_tuple(16, 64, 2),
    std::make_tuple(16, 16, 2),
  };
  // clang-format on

  if (!std::any_of(validWaveGemmSize.cbegin(), validWaveGemmSize.cend(),
                   [param](const auto it) noexcept -> bool {
                     int validMPerWave, validNPerWave, validKPerWave;
                     std::tie(validMPerWave, validNPerWave, validKPerWave) = it;
                     return (param.gemmMPerWave == validMPerWave) &&
                            (param.gemmNPerWave == validNPerWave) &&
                            (param.gemmKPerBlock % validKPerWave == 0);
                   }))
    return failure();

  if (blockSize < waveSize || blockSize > 4 * waveSize)
    return failure();

  if ((param.gemmMPerBlock % param.gemmMPerWave) != 0)
    return failure();

  if ((param.gemmNPerBlock % param.gemmNPerWave) != 0)
    return failure();

  // Sledgehammer hotfix because not unrolling sometimes makes the register
  // allocator break. This should be refined quickly.
  if (param.gemmAThreadCopyMoreGemmK == false) {
    return failure();
  }

  // Reject invalid KPACK values.
  return success();
}

std::vector<InitParamsAccel>
PopulateParamsWmma::getTuningParameters(KernelType opType, Type dataTypeA,
                                        Type dataTypeB, StringRef arch) const {
  ArrayRef<InitParamsAccel> params;
  std::vector<InitParamsAccel> res;
  switch (dataTypeA.getIntOrFloatBitWidth()) {
  case 8:
    params = {initParametersForward8Bit, nInitParametersForward8Bit};
    break;
  case 16:
    params = {initParametersFp16, nInitParametersFp16};
    break;
  default:
    return res;
  }
  // Only return valid Wmma params
  const int64_t waveSize = mlir::rock::lookupArchInfo(arch).waveSize;
  std::copy_if(
      params.begin(), params.end(), std::back_inserter(res),
      [&](const InitParamsAccel &param) {
        auto maybeWmmaInsn =
            WmmaInsn::select(dataTypeA, dataTypeB, waveSize, param.gemmMPerWave,
                             param.gemmNPerWave);
        if (failed(maybeWmmaInsn)) {
          return false;
        }
        WmmaInsn wmmaInsn = *maybeWmmaInsn;
        if (!wmmaInsn.isCoherentWithK(param.gemmKPack, param.gemmKPerBlock)) {
          return false;
        }
        return true;
      });
  return res;
}

Attribute
PopulateParamsWmma::getGemmParamsAttr(OpBuilder builder,
                                      InitParamsAccel validParams) const {
  return builder.getAttr<WmmaGemmParamsAttr>(
      validParams.gemmKPerBlock, validParams.gemmMPerBlock,
      validParams.gemmNPerBlock, validParams.gemmKPack,
      validParams.gemmMPerWave, validParams.gemmNPerWave,
      validParams.gemmAThreadCopyMoreGemmK);
}
