#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/ConvolutionDims.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/MfmaInsnGroup.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/IR/WmmaInsnGroup.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
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

/// Non-xdlops
// clang-format off
#define NonAccel_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef NonAccel_DEFINITIONS_GEN
// clang-format on

PopulateParamsInfo PopulateParamsInfo::fromOp(RockGemmWrapperInterface op) {
  PopulateParamsInfo info{op.getGemmSize(),      rock::getArchValue(op),
                          rock::getFeatures(op), op.getAType(),
                          op.getBType(),         op.getKernelType()};

  if (auto convOp = dyn_cast<ConvBwdWeightOp>(*op)) {
    auto convDims = ConvolutionDims::fromOp(op);
    info.numCu = rock::getNumCUValue(convOp);
    info.batchSize = convDims.n;
  }
  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  WalkResult wRes = func.walk(
      [&](ReduceOp rOp) -> WalkResult { return WalkResult::interrupt(); });
  info.hasFusedReduction = wRes.wasInterrupted();
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

GemmSize mlir::rock::calculatePaddedGemmSize(int64_t kPerBlock,
                                             int64_t mPerBlock,
                                             int64_t nPerBlock,
                                             GemmSize gemmSize, int64_t kPack) {
  auto gemmExtraPad =
      calculatePadding(kPerBlock, mPerBlock, nPerBlock, gemmSize, kPack);

  if (gemmExtraPad.has_value()) {
    gemmSize.m += gemmExtraPad->m;
    gemmSize.k += gemmExtraPad->k;
    gemmSize.n += gemmExtraPad->n;
  }
  return gemmSize;
}

std::optional<GemmSize> mlir::rock::requiredPadding(Attribute params,
                                                    GemmSize gemmSize,
                                                    int64_t mulByKPerBlock,
                                                    int64_t mulByMPerBlock,
                                                    int64_t mulByNPerBlock) {
  int64_t kPerBlock, mPerBlock, nPerBlock;
  int64_t kPack = 1;
  if (auto generalParams = dyn_cast<GeneralGemmParamsAttr>(params)) {
    kPerBlock = generalParams.getKPerBlock();
    mPerBlock = generalParams.getMPerBlock();
    nPerBlock = generalParams.getNPerBlock();
  } else if (auto accelParams =
                 dyn_cast<RockAccelTuningParamAttrInterface>(params)) {
    kPerBlock = accelParams.getKpackPerBlock();
    mPerBlock = accelParams.getMPerBlock();
    nPerBlock = accelParams.getNPerBlock();
    kPack = accelParams.getKpack();
  } else {
    llvm_unreachable("The tuning parameters are general or xdlops");
  }
  return calculatePadding(kPerBlock * mulByKPerBlock,
                          mPerBlock * mulByMPerBlock,
                          nPerBlock * mulByNPerBlock, gemmSize, kPack);
}

int64_t mlir::rock::obtainBlockSize(int64_t waveSize, int64_t mPerBlock,
                                    int64_t nPerBlock, int64_t mPerWave,
                                    int64_t nPerWave) {
  return waveSize * (mPerBlock / mPerWave) * (nPerBlock / nPerWave);
}

int64_t mlir::rock::obtainBlockSize(int64_t waveSize,
                                    RockAccelTuningParamAttrInterface params) {
  return obtainBlockSize(waveSize, params.getMPerBlock(), params.getNPerBlock(),
                         params.getMPerWave(), params.getNPerWave());
}

LogicalResult PopulateParams::calculateBlockGemmPerformanceParameters(
    GeneralGemmParamsAttr params) {

  FailureOr<GeneralGemmBlockStructure> maybeDerived =
      deriveGeneralGemmBlockStructure(params.getBlockSize());
  if (failed(maybeDerived))
    return failure();
  GeneralGemmBlockStructure derived = *maybeDerived;

  if (params.getMPerThread() < 2 || params.getMPerThread() > 4)
    return failure();

  if (params.getNPerThread() < 2 || params.getNPerThread() > 4)
    return failure();

  if (params.getMPerBlock() % params.getMPerThread() != 0 ||
      params.getNPerBlock() % params.getNPerThread() != 0)
    return failure();

  int64_t threadGemmMPerCluster = params.getMPerThread() *
                                  derived.mThreadsPerCuwave *
                                  derived.mCuwavesPerBlock;
  int64_t threadGemmNPerCluster = params.getNPerThread() *
                                  derived.nThreadsPerCuwave *
                                  derived.nCuwavesPerBlock;

  if ((params.getMPerBlock() % threadGemmMPerCluster != 0) ||
      (params.getNPerBlock() % threadGemmNPerCluster != 0)) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "M per block or N per block aren't divisible by M/N per cluster\n");
    return failure();
  }

  return success();
}

LogicalResult PopulateParams::populateDerived(GeneralGemmParamsAttr params) {
  LogicalResult res = calculateBlockGemmPerformanceParameters(params);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent blockGemm tuning parameter "
                            << " size.\n");
    return failure();
  }

  return success();
}

LogicalResult
PopulateParams::paramsProbablyValid(OpBuilder &b,
                                    const PopulateParamsInfo &info,
                                    GeneralGemmParamsAttr params) {
  return populateDerived(params);
}

static LogicalResult couldFusedReductionBePerformant(const GemmSize &gemmSize,
                                                     int64_t mPerBlock,
                                                     int64_t nPerBlock) {
  // 16 is practically lowest m in MFMAs/WMMAs
  // that could be performant. If the gemm sizes
  // are not divisible by that, then we definitely
  // need padding. Therefore, it can't use blockwise
  // reductions.

  // Thus, it becomes a competition among
  // atomic_store based reduction kernels.
  // So basically, all configs could be performant relative to each other.
  if (gemmSize.m % 16 != 0) {
    return success();
  }
  if (gemmSize.n % 16 != 0) {
    return success();
  }
  // We can skip knowing that dPerBlock=16
  // is there on the tuning space that should
  // be faster than anyone that use m or n
  // padding.
  if (gemmSize.m % mPerBlock != 0) {
    return failure();
  }
  if (gemmSize.n % nPerBlock != 0) {
    return failure();
  }
  return success();
}

LogicalResult PopulateParams::couldBePerformant(const PopulateParamsInfo &info,
                                                GeneralGemmParamsAttr params) {
  if (info.hasFusedReduction) {
    return couldFusedReductionBePerformant(info.gemmSize, params.getMPerBlock(),
                                           params.getNPerBlock());
  }
  return success();
}

LogicalResult PopulateParams::obtainTuningParameters(
    OpBuilder &b, const PopulateParamsInfo &info, const StringRef perfConfig,
    GeneralGemmParamsAttr &validParams) {

  if (!perfConfig.empty()) {
    // Under two scenarios can we receive a perfConfig:
    // 1. This is tuning mode
    // 2. This is running mode and we have succeeded with a perfdb load
    auto perfConfigAttr = StringAttr::get(b.getContext(), perfConfig);
    auto parsedParams = GeneralGemmParamsAttr::get(perfConfigAttr);
    if (parsedParams) {
      validParams = parsedParams;
      LLVM_DEBUG(llvm::dbgs() << validParams << "\n");
      return populateDerived(validParams);
    }
    // Signal the client if perfConfig is passed in but is invalid
    return failure();
  }

  // Backup path: Use the set of default tuning parameters
  LogicalResult res = failure();
  auto paramSets = getTuningParameters(b, info.kernelType, info.gemmAType,
                                       info.gemmBType, info.arch);
  for (auto &params : orderParams(paramSets, info.gemmSize)) {
    res = populateDerived(params);
    if (failed(res)) {
      continue;
    }

    validParams = params;
    break;
  }
  LLVM_DEBUG(llvm::dbgs() << validParams << "\n");

  return res;
}

LogicalResult PopulateParams::obtainTuningParameters(
    OpBuilder &b, RockGemmWrapperInterface op, const StringRef perfConfig,
    GeneralGemmParamsAttr &validParams) {
  PopulateParamsInfo info = PopulateParamsInfo::fromOp(op);
  return obtainTuningParameters(b, info, perfConfig, validParams);
}

std::vector<GeneralGemmParamsAttr>
PopulateParams::getTuningParameters(OpBuilder &b, KernelType opType,
                                    Type dataTypeA, Type dataTypeB,
                                    StringRef arch) const {
  auto perfConfigs =
      ParamLookupTable<GeneralGemmParamsAttr>::lookup(arch, opType, dataTypeA);
  std::vector<GeneralGemmParamsAttr> result;
  result.reserve(perfConfigs.size());
  for (StringRef perfConfig : perfConfigs) {
    auto perfConfigAttr = StringAttr::get(b.getContext(), perfConfig);
    if (auto params = GeneralGemmParamsAttr::get(perfConfigAttr)) {
      result.push_back(params);
    }
  }
  return result;
}

static int64_t calculatePaddingComplexity(const GemmSize &paddingAmount,
                                          const GemmSize &gemmSize) {
  int64_t nonPaddedComplexity = gemmSize.m * gemmSize.k * gemmSize.n;
  int64_t paddedComplexity = (gemmSize.m + paddingAmount.m) *
                             (gemmSize.k + paddingAmount.k) *
                             (gemmSize.n + paddingAmount.n);
  return paddedComplexity - nonPaddedComplexity;
}

int64_t PopulateParams::calculatePaddingAmount(GeneralGemmParamsAttr params,
                                               const GemmSize &gemmSize) const {
  std::optional<GemmSize> maybeGemmExtraPad =
      calculatePadding(params.getKPerBlock(), params.getMPerBlock(),
                       params.getNPerBlock(), gemmSize);
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
PopulateParamsAccel::calculatePaddingAmount(AccelGemmParamsAttr params,
                                            const GemmSize &gemmSize) const {
  std::optional<GemmSize> maybeGemmExtraPad =
      calculatePadding(params.getKpackPerBlock(), params.getMPerBlock(),
                       params.getNPerBlock(), gemmSize, params.getKpack());
  if (maybeGemmExtraPad.has_value()) {
    return calculatePaddingComplexity(maybeGemmExtraPad.value(), gemmSize);
  }
  return 0;
}

LogicalResult PopulateParamsAccel::paramsProbablyValid(
    OpBuilder &b, const PopulateParamsInfo &info, AccelGemmParamsAttr params) {
  RockAccelTuningParamAttrInterface accelParams =
      cast<RockAccelTuningParamAttrInterface>(params);
  return isValidBlockwiseGemm(accelParams, info.gemmAType, info.gemmBType,
                              info.arch);
}

LogicalResult
PopulateParamsAccel::couldBePerformant(const PopulateParamsInfo &info,
                                       AccelGemmParamsAttr params) {
  if (info.hasFusedReduction) {
    return couldFusedReductionBePerformant(info.gemmSize, params.getMPerBlock(),
                                           params.getNPerBlock());
  }

  return specificCouldBePerformant(params, info.gemmAType, info.gemmBType);
}

LogicalResult PopulateParamsAccel::obtainTuningParameters(
    OpBuilder &b, const PopulateParamsInfo &info, const StringRef perfConfig,
    AccelGemmParamsAttr &validParams) {

  if (!perfConfig.empty()) {
    // Under two scenarios can we receive a perfConfig:
    // 1. This is tuning mode
    // 2. This is running mode and we have succeeded with a perfdb load
    bool isWmma = bitEnumContainsAll(info.gemmFeatures, GemmFeatures::wmma);
    auto perfConfigAttr = StringAttr::get(b.getContext(), perfConfig);
    auto parsedParams = AccelGemmParamsAttr::get(perfConfigAttr, isWmma);
    if (parsedParams) {
      validParams = parsedParams;
      LLVM_DEBUG(llvm::dbgs() << validParams << "\n");
      return paramsProbablyValid(b, info, validParams);
    }
    // Signal the client if perfConfig is passed in but is invalid
    return failure();
  }

  LogicalResult res = failure();
  auto paramSets = getTuningParameters(b, info.kernelType, info.gemmAType,
                                       info.gemmBType, info.arch);

  for (const auto &params : orderParams(paramSets, info.gemmSize)) {
    res = paramsProbablyValid(b, info, params);
    if (failed(res)) {
      continue;
    }
    validParams = params;
    break;
  }
  LLVM_DEBUG(llvm::dbgs() << validParams << "\n");
  return res;
}

LogicalResult PopulateParamsAccel::obtainTuningParameters(
    OpBuilder &b, RockGemmWrapperInterface op, const StringRef perfConfig,
    AccelGemmParamsAttr &validParams) {
  PopulateParamsInfo info = PopulateParamsInfo::fromOp(op);
  auto res = obtainTuningParameters(b, info, perfConfig, validParams);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Couldn't pick heuristic values for ");
    LLVM_DEBUG(op->print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
  return res;
}

/// Xdlops acceleration
// clang-format off
#define XDL_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef XDL_DEFINITIONS_GEN
// clang-format on

LogicalResult
PopulateParamsXDL::isValidBlockwiseGemm(RockAccelTuningParamAttrInterface param,
                                        Type dataTypeA, Type dataTypeB,
                                        StringRef arch) {

  const int64_t waveSize = mlir::rock::lookupArchInfo(arch).waveSize;
  int64_t blockSize = obtainBlockSize(waveSize, param);
  if (blockSize > maxHardwareWorkgroupSize)
    return failure();
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

  if (param.getMnPerXdl() > param.getMPerWave() ||
      param.getMnPerXdl() > param.getNPerWave()) {
    LLVM_DEBUG(llvm::dbgs() << "mnPerXdl is too large:" << param << "\n");
    return failure();
  }

  // Add broadcasts for non 8-bit types.
  bool is8BitReduceOnly = dataTypeA.getIntOrFloatBitWidth() == 8;
  if (!is8BitReduceOnly) {
    validWaveGemmSize.emplace_back(8, 64, 1);
    validWaveGemmSize.emplace_back(4, 64, 1);
  }

  // Check for valid repeats and k distributions
  int64_t minDPerWave = std::min(param.getMPerWave(), param.getNPerWave());
  int64_t validKPerWaveFactor = 2;
  if (minDPerWave <= 16) {
    validKPerWaveFactor = 4;
  }
  if ((param.getMPerBlock() % minDPerWave != 0) ||
      (param.getNPerBlock() % minDPerWave != 0) ||
      ((param.getKpackPerBlock() * param.getKpack()) % validKPerWaveFactor !=
       0)) {
    return failure();
  }

  if (blockSize < waveSize) {
    return failure();
  }

  if ((param.getMPerBlock() % param.getMPerWave()) != 0) {
    return failure();
  }

  if ((param.getNPerBlock() % param.getNPerWave()) != 0) {
    return failure();
  }

  if (param.getMPerWave() % param.getMnPerXdl() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "tuning: mPerWave not divisible by mnPerXdl\n");
    return failure();
  }

  if (param.getNPerWave() % param.getMnPerXdl() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "tuning: nPerWave not divisible by mnPerXdl\n");
    return failure();
  }

  // Reject invalid blockSize
  int64_t kPerBlock = param.getKpackPerBlock() * param.getKpack();
  int64_t mPerBlock = param.getMPerBlock();
  int64_t nPerBlock = param.getNPerBlock();
  if (!isValidBlockSize(blockSize, kPerBlock, mPerBlock, nPerBlock)) {
    LLVM_DEBUG(llvm::dbgs() << "tuning: Block size too large.\n");
    return failure();
  }

  // Sledgehammer hotfix because not unrolling sometimes makes the register
  // allocator break. This should be refined quickly.
  if (!cast<RockTuningParamAttrInterface>(param).getForceUnroll()) {
    return failure();
  }

  // Reject invalid KPACK values.
  int64_t mnPerXdl = std::min(param.getMPerWave(), param.getNPerWave());
  if (auto derivedParam = dyn_cast<AccelGemmParamsAttr>(param)) {
    mnPerXdl = derivedParam.getMnPerXdl();
  }
  auto maybeMfmaInsnGroup =
      MfmaInsnGroup::select(dataTypeA, dataTypeB, arch, mnPerXdl,
                            param.getKpack(), param.getKpackPerBlock());
  if (failed(maybeMfmaInsnGroup)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to select xdlops instruction group.\n");
    return failure();
  }
  MfmaInsnGroup mfmaGroup = *maybeMfmaInsnGroup;
  if (!mfmaGroup.isCoherentWithK(param.getKpack(), param.getKpackPerBlock())) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Mfma instruction group selection is not compatible with k.\n");
    return failure();
  }

  return success();
}

std::vector<AccelGemmParamsAttr>
PopulateParamsXDL::getTuningParameters(OpBuilder &b, KernelType opType,
                                       Type dataTypeA, Type dataTypeB,
                                       StringRef arch) const {
  auto perfConfigs =
      ParamLookupTable<AccelGemmParamsAttr>::lookup(arch, opType, dataTypeA);

  std::vector<AccelGemmParamsAttr> res;
  // Only return valid XDLOp params
  for (StringRef perfConfig : perfConfigs) {
    auto perfConfigAttr = StringAttr::get(b.getContext(), perfConfig);
    auto params = AccelGemmParamsAttr::get(perfConfigAttr, /*isWmma=*/false);
    if (!params)
      continue;

    int64_t mnPerXdl = params.getMnPerXdl();
    auto maybeMfmaInsnGroup =
        MfmaInsnGroup::select(dataTypeA, dataTypeB, arch, mnPerXdl,
                              params.getKpack(), params.getKpackPerBlock());
    if (failed(maybeMfmaInsnGroup)) {
      continue;
    }
    MfmaInsnGroup mfmaGroup = *maybeMfmaInsnGroup;
    if (mfmaGroup.isCoherentWithK(params.getKpack(),
                                  params.getKpackPerBlock())) {
      res.push_back(params);
    }
  }
  return res;
}

LogicalResult
PopulateParamsXDL::specificCouldBePerformant(AccelGemmParamsAttr params,
                                             Type dataTypeA, Type dataTypeB) {

  // to keep full tuning as it was, limit numWaves <= 4
  int64_t nPerWave = params.getNPerWave();
  int64_t mWaves = params.getMPerBlock() / params.getMPerWave();
  int64_t nWaves = params.getNPerBlock() / params.getNPerWave();
  int64_t mnPerXdl = params.getMnPerXdl();
  int64_t numWaves = mWaves * nWaves;
  if ((numWaves == 4 && mnPerXdl <= nPerWave) ||
      (numWaves == 2 && mnPerXdl == nPerWave) ||
      (numWaves == 1 && mnPerXdl == nPerWave))
    return success();

  return failure();
}

/// Wmma acceleration
// clang-format off
#define Wmma_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Wmma_DEFINITIONS_GEN
// clang-format on

LogicalResult PopulateParamsWmma::isValidBlockwiseGemm(
    RockAccelTuningParamAttrInterface param, Type dataTypeA, Type dataTypeB,
    StringRef arch) {

  const int64_t waveSize = mlir::rock::lookupArchInfo(arch).waveSize;
  int64_t blockSize = obtainBlockSize(waveSize, param);
  if (blockSize > maxHardwareWorkgroupSize)
    return failure();

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

  if (param.getMnPerXdl() != 16) {
    LLVM_DEBUG(llvm::dbgs() << "mnPerXdl must be 16\n");
    return failure();
  }

  if (param.getMnPerXdl() > param.getMPerWave() ||
      param.getMnPerXdl() > param.getNPerWave()) {
    LLVM_DEBUG(llvm::dbgs() << "mnPerXdl is too large:" << param << "\n");
    return failure();
  }

  // Check for valid repeats and k distributions
  int64_t minDPerWave = std::min(param.getMPerWave(), param.getNPerWave());
  int64_t validKPerWaveFactor = 2;
  if (minDPerWave <= 16) {
    validKPerWaveFactor = 4;
  }
  if ((param.getMPerBlock() % minDPerWave != 0) ||
      (param.getNPerBlock() % minDPerWave != 0) ||
      (param.getKpackPerBlock() % validKPerWaveFactor != 0)) {
    return failure();
  }

  if (blockSize < waveSize)
    return failure();

  if ((param.getMPerBlock() % param.getMPerWave()) != 0)
    return failure();

  if ((param.getNPerBlock() % param.getNPerWave()) != 0)
    return failure();

  if (param.getMPerWave() % param.getMnPerXdl() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "tuning: mPerWave not divisible by mnPerXdl\n");
    return failure();
  }

  if (param.getNPerWave() % param.getMnPerXdl() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "tuning: nPerWave not divisible by mnPerXdl\n");
    return failure();
  }

  // Sledgehammer hotfix because not unrolling sometimes makes the register
  // allocator break. This should be refined quickly.
  if (!param.getForceUnroll()) {
    return failure();
  }

  // Reject invalid KPACK values.
  auto maybeWmmaInsn = WmmaInsn::select(
      dataTypeA, dataTypeB, waveSize, arch, param.getMPerWave(),
      param.getNPerWave(), param.getKpack(), param.getKpackPerBlock());
  if (failed(maybeWmmaInsn)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to select wmma instruction.\n");
    return failure();
  }
  WmmaInsn wmmaInsn = *maybeWmmaInsn;
  if (!wmmaInsn.isCoherentWithK(param.getKpack(), param.getKpackPerBlock())) {
    LLVM_DEBUG(llvm::dbgs()
               << "Wmma instruction selection is not compatible with k.\n");
    return failure();
  }

  return success();
}

std::vector<AccelGemmParamsAttr>
PopulateParamsWmma::getTuningParameters(OpBuilder &b, KernelType opType,
                                        Type dataTypeA, Type dataTypeB,
                                        StringRef arch) const {
  auto perfConfigs =
      ParamLookupTable<AccelGemmParamsAttr>::lookup(arch, opType, dataTypeA);

  std::vector<AccelGemmParamsAttr> res;
  // Only return valid Wmma params
  const int64_t waveSize = mlir::rock::lookupArchInfo(arch).waveSize;
  for (StringRef perfConfig : perfConfigs) {
    auto perfConfigAttr = StringAttr::get(b.getContext(), perfConfig);
    auto params = AccelGemmParamsAttr::get(perfConfigAttr, /*isWmma=*/true);
    if (!params)
      continue;

    auto maybeWmmaInsn = WmmaInsn::select(
        dataTypeA, dataTypeB, waveSize, arch, params.getMPerWave(),
        params.getNPerWave(), params.getKpack(), params.getKpackPerBlock());
    if (failed(maybeWmmaInsn)) {
      continue;
    }
    WmmaInsn wmmaInsn = *maybeWmmaInsn;
    if (wmmaInsn.isCoherentWithK(params.getKpack(),
                                 params.getKpackPerBlock())) {
      res.push_back(params);
    }
  }
  return res;
}

LogicalResult
PopulateParamsWmma::specificCouldBePerformant(AccelGemmParamsAttr params,
                                              Type dataTypeA, Type dataTypeB) {
  // Implement this if needed.
  (void)params;
  (void)dataTypeA;
  (void)dataTypeB;
  return success();
}
