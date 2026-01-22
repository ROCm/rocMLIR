#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/ConvolutionDims.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
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
  if (auto accelParams = dyn_cast<GemmParamsAttr>(params)) {
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

int64_t mlir::rock::obtainBlockSize(int64_t waveSize, GemmParamsAttr params) {
  return waveSize * params.getNumWaves();
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

static int64_t calculatePaddingComplexity(const GemmSize &paddingAmount,
                                          const GemmSize &gemmSize) {
  int64_t nonPaddedComplexity = gemmSize.m * gemmSize.k * gemmSize.n;
  int64_t paddedComplexity = (gemmSize.m + paddingAmount.m) *
                             (gemmSize.k + paddingAmount.k) *
                             (gemmSize.n + paddingAmount.n);
  return paddedComplexity - nonPaddedComplexity;
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
PopulateParamsAccel::calculatePaddingAmount(GemmParamsAttr params,
                                            const GemmSize &gemmSize) const {
  std::optional<GemmSize> maybeGemmExtraPad =
      calculatePadding(params.getKpackPerBlock(), params.getMPerBlock(),
                       params.getNPerBlock(), gemmSize, params.getKpack());
  if (maybeGemmExtraPad.has_value()) {
    return calculatePaddingComplexity(maybeGemmExtraPad.value(), gemmSize);
  }
  return 0;
}

LogicalResult
PopulateParamsAccel::couldBePerformant(const PopulateParamsInfo &info,
                                       GemmParamsAttr params) {
  if (info.hasFusedReduction) {
    return couldFusedReductionBePerformant(info.gemmSize, params.getMPerBlock(),
                                           params.getNPerBlock());
  }

  return specificCouldBePerformant(params, info.gemmAType, info.gemmBType);
}

LogicalResult PopulateParamsAccel::obtainTuningParameters(
    OpBuilder &b, const PopulateParamsInfo &info, const StringRef perfConfig,
    GemmParamsAttr &validParams) {

  if (!perfConfig.empty()) {
    // Under two scenarios can we receive a perfConfig:
    // 1. This is tuning mode
    // 2. This is running mode and we have succeeded with a perfdb load
    auto perfConfigAttr = StringAttr::get(b.getContext(), perfConfig);
    auto parsedParams = GemmParamsAttr::get(perfConfigAttr);
    if (parsedParams) {
      validParams = parsedParams;
      LLVM_DEBUG(llvm::dbgs() << validParams << "\n");
      return success();
    }
    // Signal the client if perfConfig is passed in but is invalid
    return failure();
  }

  LogicalResult res = failure();
  auto paramSets = getTuningParameters(b, info.kernelType, info.gemmAType,
                                       info.gemmBType, info.arch);

  for (const auto &params : orderParams(paramSets, info.gemmSize)) {
    validParams = params;
    break;
  }
  LLVM_DEBUG(llvm::dbgs() << validParams << "\n");
  return res;
}

LogicalResult PopulateParamsAccel::obtainTuningParameters(
    OpBuilder &b, RockGemmWrapperInterface op, const StringRef perfConfig,
    GemmParamsAttr &validParams) {
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

std::vector<GemmParamsAttr>
PopulateParamsXDL::getTuningParameters(OpBuilder &b, KernelType opType,
                                       Type dataTypeA, Type dataTypeB,
                                       StringRef arch) const {
  auto perfConfigs =
      ParamLookupTable<GemmParamsAttr>::lookup(arch, opType, dataTypeA);

  std::vector<GemmParamsAttr> res;
  for (StringRef perfConfig : perfConfigs) {
    auto perfConfigAttr = StringAttr::get(b.getContext(), perfConfig);
    auto params = GemmParamsAttr::get(perfConfigAttr);
    if (!params)
      continue;

    res.push_back(params);
  }
  return res;
}

LogicalResult
PopulateParamsXDL::specificCouldBePerformant(GemmParamsAttr params,
                                             Type dataTypeA, Type dataTypeB) {
  // Implement this if needed.
  (void)params;
  (void)dataTypeA;
  (void)dataTypeB;
  return success();
}

/// Wmma acceleration
// clang-format off
#define Wmma_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Wmma_DEFINITIONS_GEN
// clang-format on

std::vector<GemmParamsAttr>
PopulateParamsWmma::getTuningParameters(OpBuilder &b, KernelType opType,
                                        Type dataTypeA, Type dataTypeB,
                                        StringRef arch) const {
  auto perfConfigs =
      ParamLookupTable<GemmParamsAttr>::lookup(arch, opType, dataTypeA);

  std::vector<GemmParamsAttr> res;
  for (StringRef perfConfig : perfConfigs) {
    auto perfConfigAttr = StringAttr::get(b.getContext(), perfConfig);
    auto params = GemmParamsAttr::get(perfConfigAttr);
    if (!params)
      continue;

    res.push_back(params);
  }
  return res;
}

LogicalResult
PopulateParamsWmma::specificCouldBePerformant(GemmParamsAttr params,
                                              Type dataTypeA, Type dataTypeB) {
  // Implement this if needed.
  (void)params;
  (void)dataTypeA;
  (void)dataTypeB;
  return success();
}
