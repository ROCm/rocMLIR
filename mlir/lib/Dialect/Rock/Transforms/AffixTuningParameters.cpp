#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/UtilityParams.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
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
  void affixTuningParametersImpl(AttentionOp op);

  template <typename T>
  void setUtilityKernelSizes(Value arg, T utilityOp);
};
} // anonymous namespace

void AffixTuningParameters::runOnOperation() {
  func::FuncOp func = getOperation();

  func.walk(
      [&](RockGemmWrapperInterface op) { affixTuningParametersImpl(op); });
  func.walk([&](AttentionOp op) { affixTuningParametersImpl(op); });
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

  func.walk([&](GemmOp op) {
    if (op.getStoreMethod() == StoreMethod::AtomicAdd) {
      OpBuilder b(op.getContext());
      auto func = llvm::cast<func::FuncOp>(op->getParentOp());
      auto c = op.getC();
      auto attrName = mhal::PrefillAttr::getMnemonic();
      auto elementType = cast<MemRefType>(c.getType()).getElementType();
      Attribute zero;
      if (llvm::isa<FloatType>(elementType)) {
        zero = b.getFloatAttr(elementType, 0.0);
      } else {
        assert(llvm::isa<IntegerType>(elementType) &&
               "expecting `int` element type");
        zero = b.getIntegerAttr(elementType, 0);
      }
      func.setArgAttrs(2, b.getNamedAttr(attrName, zero));
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

  std::string perfConfig;
  if (auto perfConfigAttr =
          op->template getAttrOfType<StringAttr>("perf_config")) {
    perfConfig = perfConfigAttr.getValue().str();
  }

  GemmFeatures features = op.getGemmFeatures();
  if (isAccel(features)) {
    auto populateParamsAccelPtr = PopulateParamsAccel::select(features);
    InitParamsAccel validParams;
    LogicalResult status = populateParamsAccelPtr->obtainTuningParameters(
        op, perfConfig, validParams);

    if (failed(status)) {
      // Try again if allowed.
      if (fallBackNoConfig) {
        perfConfig.clear();
        status = populateParamsAccelPtr->obtainTuningParameters(op, perfConfig,
                                                                validParams);
      }
      if (failed(status)) {
        LLVM_DEBUG(llvm::dbgs() << "obtainTuningParameters call fails.\n");
        signalPassFailure();
        return;
      }
    }

    auto origGemmSize = op.getGemmSize();
    auto paddedGemmSize = calculatePaddedGemmSize(validParams, origGemmSize,
                                                  validParams.gemmKPack);
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
        signalPassFailure();
        return;
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
  } else {
    InitParamsNonAccel validParams;

    PopulateParams populateParams;
    LogicalResult status =
        populateParams.obtainTuningParameters(op, perfConfig, validParams);

    if (failed(status)) {
      signalPassFailure();
      return;
    }

    Attribute gemmParams = populateParams.getGemmParamsAttr(b, validParams);
    op.setGemmParamsAttr(gemmParams);

    // Set attributes on the function.
    getOperation()->setAttr("block_size",
                            b.getI32IntegerAttr(validParams.blockSize));
  }
}
static RockAccelTuningParamAttrInterface
deriveGemm1TuningParams(OpBuilder &builder, AttentionOp op,
                        AttnPerfConfigAttr attnPerfConfig) {
  auto gemm0TuningParams =
      cast<RockAccelTuningParamAttrInterface>(op.getParams0().value());
  int64_t gemm1KPack = gemm0TuningParams.getKpack();
  int64_t gemmNPerWaveOrMnPerXdl = gemm0TuningParams.getNPerWave();
  if (auto gemm0XdlDerivedParams =
          dyn_cast<XdlopsGemmDerivedParamsAttr>(op.getParams0().value())) {
    gemmNPerWaveOrMnPerXdl = gemm0XdlDerivedParams.getMnPerXdl();
    return XdlopsGemmDerivedParamsAttr::get(
        builder.getContext(), gemm0TuningParams.getMPerBlock() / gemm1KPack,
        attnPerfConfig.getMPerBlockG1(), gemm0XdlDerivedParams.getNPerBlock(),
        gemm0TuningParams.getKpack(),
        gemm0TuningParams.getMPerWave() * (attnPerfConfig.getMPerBlockG1() /
                                           gemm0TuningParams.getMPerBlock()),
        gemm0XdlDerivedParams.getNPerWave(),
        gemm0XdlDerivedParams.getMnPerXdl(), 1,
        gemm0XdlDerivedParams.getForceUnroll());
  }
  return WmmaGemmParamsAttr::get(
      builder.getContext(), gemm0TuningParams.getMPerBlock() / gemm1KPack,
      attnPerfConfig.getMPerBlockG1(), attnPerfConfig.getNPerBlockG0(),
      gemm0TuningParams.getKpack(),
      gemm0TuningParams.getMPerWave() *
          (attnPerfConfig.getMPerBlockG1() / gemm0TuningParams.getMPerBlock()),
      gemmNPerWaveOrMnPerXdl, 1, gemm0TuningParams.getForceUnroll());
}

void AffixTuningParameters::affixTuningParametersImpl(AttentionOp op) {
  OpBuilder builder(op.getContext());
  Type elemTypeQ = cast<MemRefType>(op.getQueries().getType()).getElementType();
  Type elemTypeK = cast<MemRefType>(op.getKeys().getType()).getElementType();
  Type elemTypeV = cast<MemRefType>(op.getValues().getType()).getElementType();
  bool isAccel = rock::isAccel(op.getFeatures());
  if (!isAccel) {
    op.emitError("Currently, attention op is only supported on GPUs "
                 "with matrix accelerator extentions");
    signalPassFailure();
    return;
  }
  Attribute params0 = op.getParams0().value_or(nullptr);
  // set a default one if params is not provided
  StringAttr perfConfigStrAttr =
      builder.getStringAttr("attn:v1:32,32,32,32,32,32,1,1");
  if (!params0) {
    if (StringAttr mayBePerfConfigStrAttr =
            dyn_cast_or_null<StringAttr>(op->getAttr("perf_config"))) {
      perfConfigStrAttr = mayBePerfConfigStrAttr;
    }
  }
  auto attnPerfConfig = AttnPerfConfigAttr::get(perfConfigStrAttr);
  if (!attnPerfConfig) {
    op.emitError("perf config string has an incorrect format.");
  }
  GemmFeatures features = op.getFeatures();
  RockAccelTuningParamAttrInterface accelParams0;
  if (bitEnumContainsAny(features, GemmFeatures::mfma)) {
    auto xdlopsParams0 = XdlopsGemmParamsAttr::get(
        builder.getContext(), attnPerfConfig.getKpackPerBlock(),
        attnPerfConfig.getMPerBlockG0(), attnPerfConfig.getNPerBlockG0(),
        attnPerfConfig.getKpack(), attnPerfConfig.getMPerWave(),
        attnPerfConfig.getMnPerXdl(), 1, attnPerfConfig.getForceUnroll());
    accelParams0 = XdlopsGemmDerivedParamsAttr::get(xdlopsParams0);
  } else {
    accelParams0 = WmmaGemmParamsAttr::get(
        builder.getContext(), attnPerfConfig.getKpackPerBlock(),
        attnPerfConfig.getMPerBlockG0(), attnPerfConfig.getNPerBlockG0(),
        attnPerfConfig.getKpack(), attnPerfConfig.getMPerWave(),
        attnPerfConfig.getMnPerXdl(), 1, attnPerfConfig.getForceUnroll());
  }
  op.setParams0Attr(accelParams0);
  if (attnPerfConfig.getMPerBlockG0() > attnPerfConfig.getMPerBlockG1()) {
    op.emitError(
        "The MPerBlockG0 should be larger or equal to getMPerBlockG1.");
    signalPassFailure();
    return;
  }
  RockAccelTuningParamAttrInterface accelParams1 =
      deriveGemm1TuningParams(builder, op, attnPerfConfig);
  op.setParams1Attr(accelParams1);
  int64_t waveSize = rock::lookupArchInfo(op.getArchAttr()).waveSize;
  int64_t blockSize = waveSize * accelParams0.getNPerBlock() *
                      accelParams0.getMPerBlock() /
                      (accelParams0.getMPerWave() * accelParams0.getNPerWave());
  auto populateParamsAccelPtr = PopulateParamsAccel::select(features);
  LLVM_DEBUG(llvm::dbgs() << "accelParams0=" << accelParams0 << "\n");
  LLVM_DEBUG(llvm::dbgs() << "accelParams1=" << accelParams1 << "\n");
  LogicalResult isValidBlockwiseGemm0 =
      populateParamsAccelPtr->isValidBlockwiseGemm(
          accelParams0, elemTypeQ, elemTypeK, op.getArch(),
          /*enableBlockSizeUpperLimit=*/false,
          /*enableDPerWaveFiltering=*/false);
  LogicalResult isValidBlockwiseGemm1 =
      populateParamsAccelPtr->isValidBlockwiseGemm(
          accelParams1, elemTypeV, elemTypeV, op.getArch(),
          /*enableBlockSizeUpperLimit=*/false,
          /*enableDPerWaveFiltering=*/false);
  if (isValidBlockwiseGemm0.failed() || isValidBlockwiseGemm1.failed()) {
    op.emitError("The provided perf config is not valid");
    signalPassFailure();
    return;
  }

  IntegerAttr blockSizeAttr = builder.getI32IntegerAttr(blockSize);
  func::FuncOp funcOp = getOperation();
  funcOp->setAttr("block_size", blockSizeAttr);
}
