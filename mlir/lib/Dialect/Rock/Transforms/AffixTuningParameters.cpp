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

  // For all ops that can take a 'features' attribute, we want to get or
  // calculate those features and then take the intersection of them and
  // apply them to the top level func. This precalculation saves us from
  // constantly needing to recompute this value at later points in the pipeline.
  SmallVector<rock::GemmFeatures> allFeatures;
  func.walk([&](Operation *op) {
    if (isa<RockGemmFeaturesInterface>(op))
      allFeatures.push_back(rock::getFeatures(op));
  });

  if (allFeatures.size() >= 1) {
    assert(std::all_of(allFeatures.begin() + 1, allFeatures.end(),
                       [&](const rock::GemmFeatures &features) {
                         return features == allFeatures[0];
                       }) &&
           "All features in func should be identical");
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
  funcOp->setAttr(rock::BlockSizeAttr::getMnemonic(), blockSizeAttr);
  funcOp->setAttr(rock::GridSizeAttr::getMnemonic(), gridSizeAttr);
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

  GemmFeatures features = rock::getFeatures(op);
  if (isAccel(features)) {
    auto populateParamsAccelPtr = PopulateParamsAccel::select(features);
    GemmParamsAttr validParams;
    LogicalResult status = populateParamsAccelPtr->obtainTuningParameters(
        b, op, perfConfig, validParams);

    if (failed(status)) {
      // Try again if allowed.
      if (fallBackNoConfig) {
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
    if (maybeWrwOp &&
        isWrWAtomicKernel(info.gemmFeatures, info.gemmAType, requiredPadding)) {
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
    GemmParamsAttr gemmParams = cast<GemmParamsAttr>(validParams);
    int64_t blockSize = obtainBlockSize(waveSize, gemmParams);
    op.setGemmParamsAttr(gemmParams);

    // Set attributes on the function.
    getOperation()->setAttr(rock::BlockSizeAttr::getMnemonic(),
                            b.getI32IntegerAttr(blockSize));
  } else {
    // TODO(roctriton): fix this
    // GeneralGemmParamsAttr validParams;
    // PopulateParams populateParams;
    // LogicalResult status =
    //     populateParams.obtainTuningParameters(b, op, perfConfig,
    //     validParams);
    // if (failed(status)) {
    //   return signalPassFailure();
    // }

    // op.setGemmParamsAttr(validParams);

    // // Set attributes on the function.
    // getOperation()->setAttr("block_size",
    //                         b.getI32IntegerAttr(validParams.getBlockSize()));
  }
  // check for fusion legality with SplitK for both accel and non-accel path
  // this check should happen after perfConfig is picked either through
  // heuristics or user provided
  if (rock::isSplitKRequested(rock::getFeatures(op),
                              b.getStringAttr(perfConfig))) {
    if (failed(testFusionLegalitySplitK(funcParent))) {
      op->emitError("Fusion with SplitK perfConfig is not legal");
      return signalPassFailure();
    }
  }
}

void AffixTuningParameters::affixTuningParametersImpl(
    RockGemmGemmWrapperInterface op) {
  OpBuilder builder(op.getContext());
  bool isAccel = rock::isAccel(rock::getFeatures(op));
  if (!isAccel) {
    op.emitError("Currently, attention/gemm+gemm/conv+gemm op is only "
                 "supported on GPUs "
                 "with matrix accelerator extentions");
    return signalPassFailure();
  }
  auto funcParent = op->getParentOfType<func::FuncOp>();

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
  auto attnPerfConfig = GemmGemmParamsAttr::get(perfConfigStrAttr);
  if (!attnPerfConfig) {
    op.emitError("perf config string has an incorrect format.");
    return signalPassFailure();
  }

  auto accelParams =
      PopulateParamsGemmGemm::getGemmParams(builder, op, attnPerfConfig);
  if (failed(accelParams)) {
    op.emitError("The provided perf config is not valid");
    return signalPassFailure();
  }
  // check for splitK legality
  if (rock::isSplitKRequested(rock::getFeatures(op), perfConfigStrAttr)) {
    if (failed(testFusionLegalitySplitK(funcParent))) {
      op->emitError("Fusion with SplitK perfConfig is not legal");
      return signalPassFailure();
    }
  }
  GemmParamsAttr accelParams0, accelParams1;
  accelParams0 = accelParams->first;
  accelParams1 = accelParams->second;
  LLVM_DEBUG(llvm::dbgs() << "accelParams0=" << accelParams0 << "\n");
  LLVM_DEBUG(llvm::dbgs() << "accelParams1=" << accelParams1 << "\n");
  op.setGemm0ParamsAttr(accelParams0);
  op.setGemm1ParamsAttr(accelParams1);
}
