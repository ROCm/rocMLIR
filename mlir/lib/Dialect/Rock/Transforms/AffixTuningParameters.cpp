#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/Tuning/UtilityParams.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKAFFIXTUNINGPARAMETERSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-affix-params"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct AffixTuningParameters
    : public rock::impl::RockAffixTuningParametersPassBase<
          AffixTuningParameters> {
public:
  using rock::impl::RockAffixTuningParametersPassBase<
      AffixTuningParameters>::RockAffixTuningParametersPassBase;
  void runOnOperation() override;

private:
  // Actual implementation.
  void affixTuningParametersImpl(RockGemmWrapperInterface op);
  void affixTuningParametersImpl(RockGemmGemmWrapperInterface op);

  template <typename T>
  void setUtilityKernelSizes(Value arg, T utilityOp);
};
} // anonymous namespace

static FailureOr<std::optional<int64_t>> getScheduleVersion(func::FuncOp funcOp,
                                                            Operation *op) {
  auto scheduleVersionAttrName = rock::ScheduleVersionAttr::getMnemonic();

  std::optional<int64_t> scheduleVersion = std::nullopt;
  bool hasPerfConfig = op->hasAttrOfType<StringAttr>("perf_config");
  if (funcOp->hasAttrOfType<rock::ScheduleVersionAttr>(
          scheduleVersionAttrName) &&
      hasPerfConfig) {
    return op->emitError(
        "kernel has both perf_config and schedule_version attribute "
        "set. Please modify schedule version directly inside "
        "perf_config and remove schedule_version\n");
  }
  if (funcOp->hasAttrOfType<rock::ScheduleVersionAttr>(
          scheduleVersionAttrName)) {
    scheduleVersion = dyn_cast<rock::ScheduleVersionAttr>(
                          funcOp->removeAttr(scheduleVersionAttrName))
                          .getScheduleVersion();
  } else if (!hasPerfConfig) {
    // set default schedule
    scheduleVersion = static_cast<int64_t>(GemmLoadTileType::Default);
  }

  // check scheduleVersion is valid
  if (scheduleVersion.has_value()) {
    std::optional<GemmLoadTileType> maybeLoadType =
        rock::symbolizeGemmLoadTileType(scheduleVersion.value());
    if (!maybeLoadType.has_value())
      return op->emitOpError("schedule version value is incorrect");
  }

  return scheduleVersion;
}

void AffixTuningParameters::runOnOperation() {
  func::FuncOp func = getOperation();
  // currently, in rocMLIR we only support one Fusion Root per function.
  // Therefore we check for that here. Note that rocMLIR does generate multiple
  // kernels for the conv_bwd_data but that decomposition happens later in the
  // pipeline.
  uint32_t fusionRootCnt = 0;
  func.walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::rock::FusionRoot>()) {
      fusionRootCnt++;
    }
  });
  if (fusionRootCnt > 1) {
    func.emitError("Multiple Fusion Roots detected in a single "
                   "function. This is not supported.");
    signalPassFailure();
    return;
  }
  func.walk(
      [&](RockGemmWrapperInterface op) { affixTuningParametersImpl(op); });
  func.walk(
      [&](RockGemmGemmWrapperInterface op) { affixTuningParametersImpl(op); });
  func.walk([&](ReduceOp op) {
    func::FuncOp funcOp = getOperation();
    if (!funcOp->hasAttr("block_size")) {
      funcOp->setAttr("block_size", op.getBlockSizeAttr());
      funcOp->setAttr("grid_size", op.getGridSizeAttr());
    }
  });
  func.walk([&](ConvertingCopyKernelOp op) {
    setUtilityKernelSizes(op.getInput(), op);
  });

  // For all ops that can take a 'features' attribute, we want to get or
  // calculate those features and then take the intersection of them and
  // apply them to the top level func. This precalculation saves us from
  // constantly needing to recompute this value at later points in the pipeline.
  SmallVector<rock::GemmFeatures> allFeatures;
  func.walk([&](Operation *op) {
    if (isa<RockGemmFeaturesInterface>(op)) {
      StringAttr arch = rock::getArchValue(op);
      rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
      SmallVector<Type> types;
      if (auto gemmOp = dyn_cast<RockGemmWrapperInterface>(op)) {
        types = {gemmOp.getAType(), gemmOp.getBType()};
      } else if (auto gemmGemmOp = dyn_cast<RockGemmGemmWrapperInterface>(op)) {
        types = {gemmGemmOp.getAType(), gemmGemmOp.getBType()};
      } else if (auto opWithFeatures =
                     dyn_cast<RockGemmFeaturesInterface>(op)) {
        types = opWithFeatures.getTypesForFeature();
      }
      allFeatures.push_back(archInfo.defaultFeatures);
    }
  });

  if (allFeatures.size() >= 1) {
    assert(std::all_of(allFeatures.begin() + 1, allFeatures.end(),
                       [&](const rock::GemmFeatures &features) {
                         return features == allFeatures[0];
                       }) &&
           "All features in func should be identical");
    func->setAttr("features",
                  rock::GemmFeaturesAttr::get(&getContext(), allFeatures[0]));
  }
}

template <typename T>
void AffixTuningParameters::setUtilityKernelSizes(Value arg, T utilityOp) {
  OpBuilder b(&getContext());

  int64_t numElements = cast<ShapedType>(arg.getType()).getNumElements();
  uint32_t blockSize = kUtilityKernelBlockSize;
  int64_t elemsPerThread = kUtilityKernelElemsPerThread;
  uint32_t gridSize =
      math_util::integer_divide_ceil(numElements, blockSize * elemsPerThread);

  IntegerAttr blockSizeAttr = b.getI32IntegerAttr(blockSize);
  IntegerAttr gridSizeAttr = b.getI32IntegerAttr(gridSize);

  // Tracking utility kernel block size separately.
  utilityOp->setAttr("blockSize", blockSizeAttr);
  utilityOp->setAttr("gridSize", gridSizeAttr);
  utilityOp->setAttr("elemsPerThread", b.getIndexAttr(elemsPerThread));

  func::FuncOp funcOp = getOperation();
  funcOp->setAttr("block_size", blockSizeAttr);
  funcOp->setAttr("grid_size", gridSizeAttr);
}

void AffixTuningParameters::affixTuningParametersImpl(
    RockGemmWrapperInterface op) {
  OpBuilder b(op.getContext());
  auto funcParent = op->getParentOfType<func::FuncOp>();
  std::string perfConfig;
  if (auto perfConfigAttr =
          op->template getAttrOfType<StringAttr>("perf_config")) {
    perfConfig = perfConfigAttr.getValue().str();
  }
  FailureOr<std::optional<int64_t>> maybeScheduleVersion =
      getScheduleVersion(funcParent, op);
  if (failed(maybeScheduleVersion))
    return signalPassFailure();

  std::optional<int64_t> scheduleVersion = maybeScheduleVersion.value();

  StringRef archStr = rock::getArchValue(op);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(archStr);
  if (archInfo.isAccel(op)) {
    GemmFeatures features = archInfo.defaultFeatures;
    auto populateParamsAccelPtr = PopulateParamsAccel::select(features);
    AccelGemmParamsAttr validParams;
    LogicalResult status = populateParamsAccelPtr->obtainTuningParameters(
        b, op, perfConfig, validParams);
    // update schedule version to what is provided by the user if and only if
    // user hasn't provided perfConfig, otherwise just keep whatever is inside
    // perfConfig
    if (scheduleVersion.has_value())
      validParams = validParams.withScheduleVersion(scheduleVersion.value());

    SmallVector<Type> types = {op.getAType(), op.getBType()};
    if (failed(isScheduleVersionSupported(validParams.getScheduleVersion(),
                                          archInfo, types, archStr))) {
      op->emitError("schedule version not supported\n");
      return signalPassFailure();
    }

    if (failed(status)) {
      // Try again if allowed.
      if (fallBackNoConfig) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "Calling obtainTuningParameters fallBack with no perf config\n");
        perfConfig.clear();
        status = populateParamsAccelPtr->obtainTuningParameters(
            b, op, perfConfig, validParams);
      }
      if (failed(status)) {
        LLVM_DEBUG(llvm::dbgs() << "obtainTuningParameters call fails.\n");
        return signalPassFailure();
      }
    }

    auto origGemmSize = op.getGemmSize();
    auto paddedGemmSize = calculatePaddedGemmSize(
        validParams.getKpackPerBlock(), validParams.getMPerBlock(),
        validParams.getNPerBlock(), origGemmSize, validParams.getKpack());
    const bool requiredPadding = !(paddedGemmSize == origGemmSize);

    int64_t gemmKBlocks = 1;
    PopulateParamsInfo info = PopulateParamsInfo::fromOp(op);
    auto maybeWrwOp = (info.kernelType == KernelType::ConvBwdWeight);
    // I dont like this hack.
    GemmFeaturesAttr featuresAttr =
        GemmFeaturesAttr::get(op.getContext(), info.gemmFeatures);
    if (maybeWrwOp && archInfo.isWrWAtomicKernel(featuresAttr, info.gemmAType,
                                                 requiredPadding)) {
      auto res = calculateKBlockNum(
          info.batchSize, paddedGemmSize, validParams.getMPerBlock(),
          validParams.getNPerBlock(), validParams.getKpackPerBlock(),
          validParams.getKpack(), info.numCu, gemmKBlocks);

      if (failed(res)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Invalid tuning parameters for computing KBlocks.\n");
        return signalPassFailure();
      }
    }

    // Set kblocks attribute only for backward weight convolutions.
    if (auto bwdOp = dyn_cast<ConvBwdWeightOp>(op.getOperation()))
      bwdOp->setAttr(bwdOp.getKBlocksAttrName(), b.getIndexAttr(gemmKBlocks));

    int64_t waveSize = rock::lookupArchInfo(rock::getArchValue(op)).waveSize;
    RockAccelTuningParamAttrInterface gemmParams =
        cast<RockAccelTuningParamAttrInterface>(validParams);
    int64_t blockSize = obtainBlockSize(waveSize, gemmParams);
    op.setDerivedBlockSizeAttr(b.getI32IntegerAttr(blockSize));
    op.setGemmParamsAttr(gemmParams);

    // Set attributes on the function.
    getOperation()->setAttr("block_size", b.getI32IntegerAttr(blockSize));
  } else {
    GeneralGemmParamsAttr validParams;
    PopulateParams populateParams;
    LogicalResult status =
        populateParams.obtainTuningParameters(b, op, perfConfig, validParams);
    if (failed(status)) {
      return signalPassFailure();
    }
    // update schedule version to what is provided by the user if and only if
    // user hasn't provided perfConfig, otherwise just keep whatever was
    // obtained from perfConfig
    if (scheduleVersion.has_value())
      validParams = validParams.withScheduleVersion(scheduleVersion.value());

    op.setGemmParamsAttr(validParams);

    // Set attributes on the function.
    getOperation()->setAttr("block_size",
                            b.getI32IntegerAttr(validParams.getBlockSize()));
  }
  // check for fusion legality with SplitK for both accel and non-accel path
  // this check should happen after perfConfig is picked either through
  // heuristics or user provided
  StringAttr arch = rock::getArchValue(op);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  GemmFeatures features = archInfo.defaultFeatures;
  if (rock::isSplitKRequested(features, b.getStringAttr(perfConfig))) {
    if (failed(testFusionLegalitySplitK(funcParent))) {
      op->emitError("Fusion with SplitK perfConfig is not legal");
      return signalPassFailure();
    }
  }
}

void AffixTuningParameters::affixTuningParametersImpl(
    RockGemmGemmWrapperInterface op) {
  OpBuilder builder(op.getContext());
  StringAttr arch = rock::getArchValue(op);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  bool isAccel = archInfo.isAccel(op);
  if (!isAccel) {
    op.emitError("Currently, attention/gemm+gemm/conv+gemm op is only "
                 "supported on GPUs "
                 "with matrix accelerator extentions");
    return signalPassFailure();
  }
  auto funcParent = op->getParentOfType<func::FuncOp>();
  FailureOr<std::optional<int64_t>> maybeScheduleVersion =
      getScheduleVersion(funcParent, op);
  if (failed(maybeScheduleVersion))
    return signalPassFailure();

  std::optional<int64_t> scheduleVersion = maybeScheduleVersion.value();

  Attribute params0 = op.getGemm0Params().value_or(nullptr);
  // set a default one if params is not provided
  StringAttr perfConfigStrAttr =
      builder.getStringAttr("attn:v3:32,32,32,32,32,32,16,1,1,1,2,0,1");
  if (!params0) {
    if (StringAttr mayBePerfConfigStrAttr =
            dyn_cast_or_null<StringAttr>(op->getAttr("perf_config"))) {
      perfConfigStrAttr = mayBePerfConfigStrAttr;
    }
  }
  bool isWmma = archInfo.isWmma(op);
  auto attnPerfConfig = GemmGemmParamsAttr::get(perfConfigStrAttr, isWmma);
  if (!attnPerfConfig) {
    op.emitError("perf config string has an incorrect format.");
    return signalPassFailure();
  }

  if (scheduleVersion.has_value())
    attnPerfConfig =
        attnPerfConfig.withScheduleVersion(scheduleVersion.value());

  SmallVector<Type> types = {op.getAType(), op.getBType()};
  if (failed(isScheduleVersionSupported(attnPerfConfig.getScheduleVersion(),
                                        archInfo, types, arch))) {
    op->emitError("schedule version not supported\n");
    return signalPassFailure();
  }

  auto accelParams =
      PopulateParamsGemmGemm::getAccelGemmParams(builder, op, attnPerfConfig);
  if (failed(accelParams)) {
    op.emitError("The provided perf config is not valid");
    return signalPassFailure();
  }
  // check for splitK legality
  GemmFeatures features = archInfo.defaultFeatures;
  if (rock::isSplitKRequested(features, perfConfigStrAttr)) {
    if (failed(testFusionLegalitySplitK(funcParent))) {
      op->emitError("Fusion with SplitK perfConfig is not legal");
      return signalPassFailure();
    }
  }
  RockAccelTuningParamAttrInterface accelParams0, accelParams1;
  accelParams0 = accelParams->first;
  accelParams1 = accelParams->second;
  LLVM_DEBUG(llvm::dbgs() << "accelParams0=" << accelParams0 << "\n");
  LLVM_DEBUG(llvm::dbgs() << "accelParams1=" << accelParams1 << "\n");
  op.setGemm0ParamsAttr(accelParams0);
  op.setGemm1ParamsAttr(accelParams1);
  int64_t waveSize = rock::lookupArchInfo(rock::getArchValue(op)).waveSize;
  int64_t blockSize = waveSize * accelParams0.getNPerBlock() *
                      accelParams0.getMPerBlock() /
                      (accelParams0.getMPerWave() * accelParams0.getNPerWave());

  IntegerAttr blockSizeAttr = builder.getI32IntegerAttr(blockSize);
  func::FuncOp funcOp = getOperation();
  funcOp->setAttr("block_size", blockSizeAttr);
}
