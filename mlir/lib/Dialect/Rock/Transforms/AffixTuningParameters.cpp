#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/UtilityParams.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
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
  func.walk(
      [&](InitKernelOp op) { setUtilityKernelSizes(op.getBuffer(), op); });
  func.walk([&](ConvertingCopyKernelOp op) {
    setUtilityKernelSizes(op.getInput(), op);
  });

  auto &bufferDeps = getAnalysis<BufferDependencyAnalysis>();
  func.walk([&](GemmOp op) {
    if (op.getStoreMethod() == StoreMethod::AtomicAdd) {
      OpBuilder b(op.getContext());
      auto func = llvm::cast<func::FuncOp>(op->getParentOp());
      auto c = op.getC();
      auto attrName = rock::PrefillAttr::getMnemonic();
      auto elementType = cast<MemRefType>(c.getType()).getElementType();
      Attribute zero;
      if (llvm::isa<FloatType>(elementType)) {
        zero = b.getFloatAttr(elementType, 0.0);
      } else {
        assert(llvm::isa<IntegerType>(elementType) &&
               "expecting `int` element type");
        zero = b.getIntegerAttr(elementType, 0);
      }
      FailureOr<SmallVector<BlockArgument>> args =
          traceGemmOutputToArgs(c, func, bufferDeps);
      assert(succeeded(args) &&
             "can't trace the GEMM output to a kernel result");
      for (auto arg : args.value())
        func.setArgAttrs(arg.getArgNumber(), b.getNamedAttr(attrName, zero));
    }
  });
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
  auto scheduleVersionAttrName = rock::ScheduleVersionAttr::getMnemonic();
  auto funcParent = op->getParentOfType<func::FuncOp>();
  std::string perfConfig;
  if (funcParent->hasAttrOfType<rock::ScheduleVersionAttr>(
          scheduleVersionAttrName) &&
      op->hasAttrOfType<StringAttr>("perf_config")) {
    op->emitError("kernel has both perf_config and schedule_version attribute "
                  "set. Please modify schedule version directly inside "
                  "perf_config and remove schedule_version\n");
    signalPassFailure();
    return;
  }
  if (auto perfConfigAttr =
          op->template getAttrOfType<StringAttr>("perf_config")) {
    perfConfig = perfConfigAttr.getValue().str();
  }
  // by default rocMLIR selects GEMM Schedule V1
  auto scheduleVersion = 1;
  if (funcParent->hasAttrOfType<rock::ScheduleVersionAttr>(
          scheduleVersionAttrName)) {
    scheduleVersion = dyn_cast<rock::ScheduleVersionAttr>(
                          funcParent->removeAttr(scheduleVersionAttrName))
                          .getScheduleVersion();
  }

  GemmFeatures features = op.getGemmFeatures();
  auto populateParamsAccelPtr = PopulateParamsAccel::select(features);
  InitParamsAccel validParams;
  LogicalResult status = populateParamsAccelPtr->obtainTuningParameters(
      op, perfConfig, validParams);
  // update schedule version to what is provided by the user if and only if
  // user hasn't provided perfConfig, otherwise just keep whatever is inside
  // perfConfig
  if (!op->hasAttrOfType<StringAttr>("perf_config")) {
    validParams.gemmScheduleVersion = scheduleVersion;
  }
  if (failed(status)) {
    // Try again if allowed.
    if (fallBackNoConfig) {
      perfConfig.clear();
      status = populateParamsAccelPtr->obtainTuningParameters(op, perfConfig,
                                                              validParams);
    }
    if (failed(status)) {
      LLVM_DEBUG(llvm::dbgs() << "obtainTuningParameters call fails.\n");
      return signalPassFailure();
    }
  }

  auto origGemmSize = op.getGemmSize();
  auto paddedGemmSize =
      calculatePaddedGemmSize(validParams, origGemmSize, validParams.gemmKPack);
  const bool requiredPadding = !(paddedGemmSize == origGemmSize);

  int64_t gemmKBlocks = 1;
  PopulateParamsInfo info = PopulateParamsInfo::fromOp(op);
  auto maybeWrwOp = (info.kernelType == KernelType::ConvBwdWeight);
  if (maybeWrwOp &&
      isWrWAtomicKernel(info.gemmFeatures, info.gemmAType, requiredPadding)) {
    auto res = calculateKBlockNum(
        info.batchSize, paddedGemmSize, validParams.gemmMPerBlock,
        validParams.gemmNPerBlock, validParams.gemmKPerBlock,
        validParams.gemmKPack, info.numCu, gemmKBlocks);

    if (failed(res)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Invalid tuning parameters for computing KBlocks.\n");
      return signalPassFailure();
    }
  }

  // Set kblocks attribute only for backward weight convolutions.
  if (auto bwdOp = dyn_cast<ConvBwdWeightOp>(op.getOperation()))
    bwdOp->setAttr(bwdOp.getKBlocksAttrName(), b.getIndexAttr(gemmKBlocks));

  int64_t waveSize = rock::lookupArchInfo(op.getArch()).waveSize;
  RockAccelTuningParamAttrInterface gemmParams;
  Attribute gemmParamsAttr =
      populateParamsAccelPtr->getGemmParamsAttr(b, validParams);
  if (auto xdlopsParams = dyn_cast<XdlopsGemmParamsAttr>(gemmParamsAttr)) {
    gemmParams = XdlopsGemmDerivedParamsAttr::get(xdlopsParams);
  } else {
    gemmParams = cast<RockAccelTuningParamAttrInterface>(gemmParamsAttr);
  }
  int64_t blockSize = obtainBlockSize(waveSize, gemmParams);
  op.setDerivedBlockSizeAttr(b.getI32IntegerAttr(blockSize));
  op.setGemmParamsAttr(gemmParams);

  // Set attributes on the function.
  getOperation()->setAttr("block_size", b.getI32IntegerAttr(blockSize));
}

static FailureOr<RockAccelTuningParamAttrInterface> deriveGemm1TuningParams(
    OpBuilder &builder, RockGemmGemmWrapperInterface op,
    RockAccelAttentionTuningParamAttrInterface attnPerfConfig) {
  auto gemm0TuningParams =
      cast<RockAccelTuningParamAttrInterface>(op.getGemm0Params().value());
  int64_t gemm1KPack = gemm0TuningParams.getKpack();

  if (gemm0TuningParams.getMPerBlock() % gemm1KPack != 0) {
    LLVM_DEBUG(llvm::dbgs() << "gemm0TuningParams.getMPerBlock() should be "
                               "divisible by gemm1KPack\n");
    return failure();
  }
  AttnPerfConfigAttr accelAttn = dyn_cast<AttnPerfConfigAttr>(attnPerfConfig);
  if (auto gemm0XdlDerivedParams =
          dyn_cast<XdlopsGemmDerivedParamsAttr>(op.getGemm0Params().value())) {
    return (RockAccelTuningParamAttrInterface)XdlopsGemmDerivedParamsAttr::get(
        builder.getContext(), gemm0TuningParams.getMPerBlock() / gemm1KPack,
        accelAttn.getMPerBlockG1(), gemm0XdlDerivedParams.getNPerBlock(),
        gemm0TuningParams.getKpack(),
        gemm0XdlDerivedParams.getMPerWave() *
            (accelAttn.getMPerBlockG1() / gemm0TuningParams.getMPerBlock()),
        gemm0XdlDerivedParams.getNPerWave(),
        gemm0XdlDerivedParams.getMnPerXdl(), 1,
        gemm0XdlDerivedParams.getScheduleVersion(),
        gemm0XdlDerivedParams.getOutputSwizzle(),
        gemm0XdlDerivedParams.getForceUnroll());

  } else if (auto gemm0WmmaParams =
                 dyn_cast<WmmaGemmParamsAttr>(op.getGemm0Params().value())) {
    return (RockAccelTuningParamAttrInterface)WmmaGemmParamsAttr::get(
        builder.getContext(), gemm0TuningParams.getMPerBlock() / gemm1KPack,
        accelAttn.getMPerBlockG1(), accelAttn.getNPerBlockG0(),
        gemm0TuningParams.getKpack(),
        gemm0WmmaParams.getMPerWave() *
            (accelAttn.getMPerBlockG1() / gemm0TuningParams.getMPerBlock()),
        gemm0WmmaParams.getNPerWave(), 1,
        gemm0TuningParams.getScheduleVersion(),
        gemm0TuningParams.getOutputSwizzle(),
        gemm0TuningParams.getForceUnroll());
  } else {
    AttnFmaPerfConfigAttr fmaAttn =
        dyn_cast<AttnFmaPerfConfigAttr>(attnPerfConfig);
    return (RockAccelTuningParamAttrInterface)FmaGemmParamsAttr::get(
        builder.getContext(), fmaAttn.getBlockSize(), fmaAttn.getMPerBlockG1(),
        fmaAttn.getNPerBlockG0(), gemm0TuningParams.getMPerBlock() / gemm1KPack,
        fmaAttn.getKpack(), 1, fmaAttn.getScheduleVersion(), 2,
        fmaAttn.getForceUnroll());
  }
}

void AffixTuningParameters::affixTuningParametersImpl(
    RockGemmGemmWrapperInterface op) {
  OpBuilder builder(op.getContext());
  Attribute params0 = op.getGemm0Params().value_or(nullptr);
  // set a default one if params is not provided
  StringAttr perfConfigStrAttr =
      builder.getStringAttr("attn:v1:32,32,32,32,32,32,1,1");
  if (!params0) {
    if (StringAttr mayBePerfConfigStrAttr =
            dyn_cast_or_null<StringAttr>(op->getAttr("perf_config"))) {
      perfConfigStrAttr = mayBePerfConfigStrAttr;
    }
  }
  RockAccelAttentionTuningParamAttrInterface attnPerfConfig =
      AttnPerfConfigAttr::get(perfConfigStrAttr);
  if (!attnPerfConfig) {
    // try to parse the FMA perf config string
    attnPerfConfig = AttnFmaPerfConfigAttr::get(perfConfigStrAttr);
    if (!attnPerfConfig) {
      op.emitError("perf config string has an incorrect format.");
      return signalPassFailure();
    }
  }
  GemmFeatures features = op.getGemmFeatures();
  RockAccelTuningParamAttrInterface accelParams0;
  AttnPerfConfigAttr accelAttn = dyn_cast<AttnPerfConfigAttr>(attnPerfConfig);
  if (bitEnumContainsAny(features, GemmFeatures::mfma)) {
    if (!accelAttn) {
      op.emitError("attn: perf config requires hardware acceleration.");
      return signalPassFailure();
    }
    auto xdlopsParams0 = XdlopsGemmParamsAttr::get(
        builder.getContext(), accelAttn.getKpackPerBlock(),
        accelAttn.getMPerBlockG0(), accelAttn.getNPerBlockG0(),
        accelAttn.getKpack(), accelAttn.getMPerWave(), accelAttn.getMnPerXdl(),
        1, accelAttn.getScheduleVersion(), 2, accelAttn.getForceUnroll());
    accelParams0 = XdlopsGemmDerivedParamsAttr::get(xdlopsParams0);
  } else if (bitEnumContainsAny(features, GemmFeatures::wmma)) {
    if (!accelAttn) {
      op.emitError("attn: perf config requires hardware acceleration.");
      return signalPassFailure();
    }
    accelParams0 = WmmaGemmParamsAttr::get(
        builder.getContext(), accelAttn.getKpackPerBlock(),
        accelAttn.getMPerBlockG0(), accelAttn.getNPerBlockG0(),
        accelAttn.getKpack(), accelAttn.getMPerWave(), accelAttn.getMnPerXdl(),
        1, accelAttn.getScheduleVersion(), 2, accelAttn.getForceUnroll());
  } else {
    AttnFmaPerfConfigAttr fmaAttn =
        dyn_cast<AttnFmaPerfConfigAttr>(attnPerfConfig);
    if (!fmaAttn) {
      op.emitError("attn_fma: perf config not found for non-accel hardware.");
      return signalPassFailure();
    }
    accelParams0 = FmaGemmParamsAttr::get(
        builder.getContext(), fmaAttn.getBlockSize(), fmaAttn.getMPerBlockG0(),
        fmaAttn.getNPerBlockG0(), fmaAttn.getKpackPerBlock(),
        fmaAttn.getKpack(), 1, fmaAttn.getScheduleVersion(), 2,
        fmaAttn.getForceUnroll());
  }
  op.setGemm0ParamsAttr(accelParams0);
  if (attnPerfConfig.getMPerBlockG0() > attnPerfConfig.getMPerBlockG1()) {
    op.emitError(
        "The MPerBlockG0 should be larger or equal to getMPerBlockG1.");
    return signalPassFailure();
  }
  FailureOr<RockAccelTuningParamAttrInterface> maybeAccelParams1 =
      deriveGemm1TuningParams(builder, op, attnPerfConfig);

  if (failed(maybeAccelParams1)) {
    op.emitError("Couldn't derive tuning parameters for second gemm.");
    return signalPassFailure();
  }

  auto accelParams1 = maybeAccelParams1.value();
  op.setGemm1ParamsAttr(accelParams1);
  int64_t waveSize = rock::lookupArchInfo(op.getArch()).waveSize;
  int64_t blockSize = accelParams0.getBlockSize(waveSize);
  auto populateParamsAccelPtr = PopulateParamsAccel::select(features);
  LLVM_DEBUG(llvm::dbgs() << "accelParams0=" << accelParams0 << "\n");
  LLVM_DEBUG(llvm::dbgs() << "accelParams1=" << accelParams1 << "\n");
  LogicalResult isValidBlockwiseGemm0 =
      populateParamsAccelPtr->isValidBlockwiseGemm(
          accelParams0, cast<MemRefType>(op.getAType()).getElementType(),
          cast<MemRefType>(op.getBType()).getElementType(), op.getArch(),
          /*enableBlockSizeUpperLimit=*/false,
          /*enableDPerWaveFiltering=*/false);
  LogicalResult isValidBlockwiseGemm1 =
      populateParamsAccelPtr->isValidBlockwiseGemm(
          accelParams1, cast<MemRefType>(op.getCType()).getElementType(),
          cast<MemRefType>(op.getCType()).getElementType(), op.getArch(),
          /*enableBlockSizeUpperLimit=*/false,
          /*enableDPerWaveFiltering=*/false);
  if (isValidBlockwiseGemm0.failed() || isValidBlockwiseGemm1.failed()) {
    op.emitError("The provided perf config is not valid");
    return signalPassFailure();
  }

  IntegerAttr blockSizeAttr = builder.getI32IntegerAttr(blockSize);
  func::FuncOp funcOp = getOperation();
  funcOp->setAttr("block_size", blockSizeAttr);
}
