#include "mlir/Dialect/Func/IR/FuncOps.h"
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

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKAFFIXTUNINGPARAMETERSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

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
  // Block size can be set in two ways:
  // * Through the MLIR lowering pass:
  //   At this case, blockSizeOverride will be initialized to zero. Then
  //   the affix tuning parameters pass will decide on a block size.
  //   Finally, block size will be hinted back to rocmlir-driver.
  // * Through cmd option "block_size":
  //   At this case, rocmlir-driver assigns a blockSizeOverride. As
  //   a result, affix tuning parameters pass should make its decisions
  //   to generate tuning parameters based on this blockSizeOverride.
  //   This guarantess that affix tuning parameters pass generate
  //   coherent tuning parameters with the pre-set block size.

  // Actual implementation.
  void affixTuningParametersImpl(RockGemmWrapperInterface op);
  void affixTuningParametersImpl(AttentionOp op);

  template <typename T> void setUtilityKernelSizes(Value arg, T utilityOp);
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
      [&](ZeroInitKernelOp op) { setUtilityKernelSizes(op.getBuffer(), op); });
  func.walk([&](ConvertingCopyKernelOp op) {
    setUtilityKernelSizes(op.getInput(), op);
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
    uint32_t blockSize = 0;
    uint32_t gridSize = 0;
    int64_t gemmKBlocks = 1;

    LogicalResult status = populateParamsAccelPtr->obtainTuningParameters(
        op, blockSizeOverride, perfConfig, validParams, blockSize, gridSize,
        gemmKBlocks);

    if (failed(status)) {
      // Try again if allowed.
      if (fallBackNoConfig) {
        perfConfig.clear();
        status = populateParamsAccelPtr->obtainTuningParameters(
            op, blockSizeOverride, perfConfig, validParams, blockSize, gridSize,
            gemmKBlocks);
      }
      if (failed(status))
        signalPassFailure();
    }

    gridSize = gridSizeOverride ? gridSizeOverride : gridSize;
    op.setDerivedBlockSizeAttr(b.getI32IntegerAttr(blockSize));
    op.setGridSizeAttr(b.getI32IntegerAttr(gridSize));

    // Set kblocks attribute only for backward weight convolutions.
    if (auto bwdOp = dyn_cast<Conv2DBwdWeightOp>(op.getOperation()))
      bwdOp->setAttr(bwdOp.getKBlocksAttrName(), b.getIndexAttr(gemmKBlocks));

    Attribute gemmParams =
        populateParamsAccelPtr->getGemmParamsAttr(b, validParams);

    op.setGemmParamsAttr(gemmParams);
    int64_t waveSize = rock::lookupArchInfo(op.getArch()).waveSize;

    // Set attributes on the function.
    getOperation()->setAttr("block_size", b.getI32IntegerAttr(blockSize));
    getOperation()->setAttr("grid_size", b.getI32IntegerAttr(gridSize));
    getOperation()->setAttr("wave_size", b.getI32IntegerAttr(waveSize));

  } else {
    InitParamsNonAccel validParams;
    uint32_t gridSize;

    PopulateParams populateParams;
    LogicalResult status = populateParams.obtainTuningParameters(
        op, blockSizeOverride, perfConfig, validParams, gridSize);

    if (failed(status)) {
      signalPassFailure();
    }

    gridSize = gridSizeOverride ? gridSizeOverride : gridSize;
    op.setGridSizeAttr(b.getI32IntegerAttr(gridSize));

    // For non-accelerator path, do not use KPack for now.

    // kPerThread and the cuwave parameters are hardcoded, may change in a
    // different pass. Please visit
    // gridwise_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw for details

    Attribute gemmParams = populateParams.getGemmParamsAttr(b, validParams);
    op.setGemmParamsAttr(gemmParams);

    int64_t waveSize = rock::lookupArchInfo(op.getArch()).waveSize;

    // Set attributes on the function.
    getOperation()->setAttr("block_size",
                            b.getI32IntegerAttr(validParams.blockSize));
    getOperation()->setAttr("grid_size", b.getI32IntegerAttr(gridSize));
    getOperation()->setAttr("wave_size", b.getI32IntegerAttr(waveSize));
  }
}

static RockAccelTuningParamAttrInterface
deriveGemm1TuningParams(OpBuilder &builder,
                        RockAccelTuningParamAttrInterface gemm0TuningParams,
                        GemmFeatures features) {
  int64_t gemm1KPack = gemm0TuningParams.getKpack();
  return builder.getAttr<XdlopsGemmParamsAttr>(
      /*gemmKpackPerBlock=*/gemm0TuningParams.getMPerBlock() / gemm1KPack,
      /*gemmMPerBlock=*/gemm0TuningParams.getMPerBlock(),
      /*gemmNPerBlock=*/gemm0TuningParams.getNPerBlock(),
      /*gemmKPack=*/gemm1KPack,
      /*gemmMPerWave=*/gemm0TuningParams.getMPerWave(),
      /*gemmNPerWave=*/gemm0TuningParams.getNPerWave(),
      /*forceUnroll=*/gemm0TuningParams.getForceUnroll());
}

void AffixTuningParameters::affixTuningParametersImpl(AttentionOp op) {
  OpBuilder builder(op.getContext());
  Value queries = op.getQueries();
  Value keys = op.getKeys();
  Value values = op.getValues();
  Type elemType = queries.getType().cast<MemRefType>().getElementType();
  bool isAccel = rock::isAccel(op.getFeatures());
  if (!isAccel) {
    op.emitError("Currently, attention op is only supported on GPUs "
                 "with matrix accelerator extentions");
    signalPassFailure();
    return;
  }
  Attribute params0 = op.getParams0().value_or(nullptr);
  // set a default one if params is not provided
  std::string perfConfigStr = "32,32,32,32,32,1,1,1";
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
  auto accelParams0 = params0.cast<RockAccelTuningParamAttrInterface>();
  op.setParams0Attr(accelParams0);
  auto accelParams1 =
      deriveGemm1TuningParams(builder, accelParams0, op.getFeatures());
  op.setParams1Attr(accelParams1);
  int64_t waveSize = rock::lookupArchInfo(op.getArchAttr()).waveSize;
  int64_t blockSize = waveSize * accelParams0.getNPerBlock() *
                      accelParams0.getMPerBlock() /
                      (accelParams0.getMPerWave() * accelParams0.getNPerWave());
  LogicalResult isValidBlockwiseGemm =
      populateParamsAccelPtr->isValidBlockwiseGemm(
          initAccelParams, elemType, elemType, op.getArch(), blockSize);
  if (isValidBlockwiseGemm.failed()) {
    op.emitError("The provided perf config is not valid");
    signalPassFailure();
    return;
  }

  // Calculate (padded) grid size
  SmallVector<int64_t, 3> queriesShape =
      llvm::to_vector<3>(queries.getType().cast<MemRefType>().getShape());
  // Note: the gridwise ops take K x M and K x N, so Q must be transposed if
  // it's in the natural M x K form
  if (!op.getQTransposed()) {
    std::iter_swap(queriesShape.rbegin(), queriesShape.rbegin() + 1);
  }
  SmallVector<int64_t, 3> keysShape =
      llvm::to_vector<3>(keys.getType().cast<MemRefType>().getShape());
  if (op.getKTransposed()) {
    std::iter_swap(keysShape.rbegin(), keysShape.rbegin() + 1);
  }
  SmallVector<int64_t, 3> valuesShape =
      llvm::to_vector<3>(values.getType().cast<MemRefType>().getShape());
  if (op.getVTransposed()) {
    std::iter_swap(valuesShape.rbegin(), valuesShape.rbegin() + 1);
  }
  GemmSize gemm0Size(/*g=*/queriesShape[0], /*m=*/keysShape[2],
                     /*k=*/queriesShape[1],
                     /*n=*/queriesShape[2]);
  GemmSize gemm0ExtraPad =
      requiredPadding(params0, gemm0Size).value_or(GemmSize{0, 0, 0, 0});
  GemmSize gemm1Size(/*g=*/queriesShape[0], /*m=*/valuesShape[2],
                     /*k=*/valuesShape[1],
                     /*n=*/keysShape[2]);
  GemmSize gemm1ExtraPad =
      requiredPadding(accelParams1, gemm1Size).value_or(GemmSize{0, 0, 0, 0});

  int64_t gridSize =
      ((gemm0Size.n + gemm0ExtraPad.n) / accelParams0.getNPerBlock()) *
      ((gemm1Size.m + gemm1ExtraPad.m) / accelParams1.getMPerBlock()) *
      gemm0Size.g;
  IntegerAttr blockSizeAttr = builder.getI32IntegerAttr(blockSize);
  IntegerAttr gridSizeAttr = builder.getI32IntegerAttr(gridSize);
  func::FuncOp funcOp = getOperation();
  funcOp->setAttr("block_size", blockSizeAttr);
  funcOp->setAttr("grid_size", gridSizeAttr);
}
