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
      auto elementType = c.getType().cast<MemRefType>().getElementType();
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

  int64_t numElements = arg.getType().cast<ShapedType>().getNumElements();
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
    auto maybeWrwOp = (info.kernelType == KernelType::Conv2DBwdWeight);
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
    if (auto bwdOp = dyn_cast<Conv2DBwdWeightOp>(op.getOperation()))
      bwdOp->setAttr(bwdOp.getKBlocksAttrName(), b.getIndexAttr(gemmKBlocks));

    int64_t waveSize = rock::lookupArchInfo(op.getArch()).waveSize;
    RockAccelTuningParamAttrInterface gemmParams;
    Attribute gemmParamsAttr =
        populateParamsAccelPtr->getGemmParamsAttr(b, validParams);
    if (auto xdlopsParams = gemmParamsAttr.dyn_cast<XdlopsGemmParamsAttr>()) {
      gemmParams = XdlopsGemmDerivedParamsAttr::get(xdlopsParams);
    } else {
      gemmParams = gemmParamsAttr.cast<RockAccelTuningParamAttrInterface>();
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

static InitParamsAccel deriveGemm1TuningParams(OpBuilder &builder,
                                               AttentionOp op) {
  auto gemm0TuningParams =
      op.getParams0().value().cast<RockAccelTuningParamAttrInterface>();
  auto oShape = op.getOut().getType().cast<ShapedType>().getShape();
  int64_t gemm1MPerBlock = gemm0TuningParams.getMPerBlock();
  // There is a nan failure when the gemm1MPerBlock
  // is increased beyond gemm0MPerBlock when getMPerWave
  // is less than 32 (i.e. 16).
  int64_t gemm1M = op.getOTransposed() ? oShape[1] : oShape[2];
  // This is good enough heuristic for now to guard from cases where
  // head dimension exceed 1024 for i8, 512 for f16 and 256 for for f32
  Type gemm1ElemType =
      op.getValues().getType().cast<ShapedType>().getElementType();
  int64_t gemm1ElemTypeByteWidth = gemm1ElemType.getIntOrFloatBitWidth() / 8;
  int64_t gemm1MUpperBound = 1024 / gemm1ElemTypeByteWidth;
  gemm1M = math_util::integer_least_multiple(gemm1M,
                                             gemm0TuningParams.getMPerBlock());
  int64_t gemm1MPerBlockNew = std::min(gemm1M, gemm1MUpperBound);
  if (gemm0TuningParams.getMPerWave() >= 32 &&
      gemm0TuningParams.getMPerBlock() < gemm1MPerBlockNew) {
    gemm1MPerBlock = gemm1MPerBlockNew;
  }
  int64_t gemm1KPack = gemm0TuningParams.getKpack();
  int64_t gemmNPerWaveOrMnPerXdl = gemm0TuningParams.getNPerWave();
  if (auto gemm0XdlDerivedParams =
          op.getParams0().value().dyn_cast<XdlopsGemmDerivedParamsAttr>()) {
    gemmNPerWaveOrMnPerXdl = gemm0XdlDerivedParams.getMnPerXdl();
  }
  return InitParamsAccel(
      /*gemmMPerBlock=*/gemm1MPerBlock,
      /*gemmNPerBlock=*/gemm0TuningParams.getNPerBlock(),
      /*gemmKpackPerBlock=*/gemm0TuningParams.getMPerBlock() / gemm1KPack,
      /*gemmMPerWave=*/gemm0TuningParams.getMPerWave() *
          (gemm1MPerBlock / gemm0TuningParams.getMPerBlock()),
      /*gemmNPerWaveOrMnPerXdl=*/gemmNPerWaveOrMnPerXdl,
      /*gemmKPack=*/gemm1KPack,
      /*splitKFactor*/ 1,
      /*forceUnroll=*/gemm0TuningParams.getForceUnroll(), false);
}

void AffixTuningParameters::affixTuningParametersImpl(AttentionOp op) {
  OpBuilder builder(op.getContext());
  Type elemTypeQ =
      op.getQueries().getType().cast<MemRefType>().getElementType();
  Type elemTypeK = op.getKeys().getType().cast<MemRefType>().getElementType();
  Type elemTypeV = op.getValues().getType().cast<MemRefType>().getElementType();
  bool isAccel = rock::isAccel(op.getFeatures());
  if (!isAccel) {
    op.emitError("Currently, attention op is only supported on GPUs "
                 "with matrix accelerator extentions");
    signalPassFailure();
    return;
  }
  Attribute params0 = op.getParams0().value_or(nullptr);
  // set a default one if params is not provided
  std::string perfConfigStr = "v2:32,32,32,32,32,1,1,1,1";
  InitParamsAccel initAccelParams;
  if (!params0) {
    if (StringAttr perfConfigStrAttr =
            dyn_cast_or_null<StringAttr>(op->getAttr("perf_config"))) {
      perfConfigStr = perfConfigStrAttr.str();
    }
  }
  GemmFeatures features = op.getFeatures();
  auto populateParamsAccelPtr = PopulateParamsAccel::select(features);
  if (initAccelParams.deserialize(perfConfigStr)) {
    params0 =
        populateParamsAccelPtr->getGemmParamsAttr(builder, initAccelParams);
  } else {
    op.emitError("perf config string has an incorrect format.");
    signalPassFailure();
    return;
  }
  RockAccelTuningParamAttrInterface accelParams0;
  if (auto xdlopsParams0 = params0.dyn_cast<XdlopsGemmParamsAttr>()) {
    accelParams0 = XdlopsGemmDerivedParamsAttr::get(xdlopsParams0);
  } else {
    accelParams0 = params0.cast<RockAccelTuningParamAttrInterface>();
  }
  op.setParams0Attr(accelParams0);
  auto initAccelParams1 = deriveGemm1TuningParams(builder, op);
  Attribute params1 =
      populateParamsAccelPtr->getGemmParamsAttr(builder, initAccelParams1);
  RockAccelTuningParamAttrInterface accelParams1;
  if (auto xdlopsParams1 = params1.dyn_cast<XdlopsGemmParamsAttr>()) {
    accelParams1 = XdlopsGemmDerivedParamsAttr::get(xdlopsParams1);
  } else {
    accelParams1 = params1.cast<RockAccelTuningParamAttrInterface>();
  }
  op.setParams1Attr(accelParams1);
  int64_t waveSize = rock::lookupArchInfo(op.getArchAttr()).waveSize;
  int64_t blockSize = waveSize * accelParams0.getNPerBlock() *
                      accelParams0.getMPerBlock() /
                      (accelParams0.getMPerWave() * accelParams0.getNPerWave());
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
