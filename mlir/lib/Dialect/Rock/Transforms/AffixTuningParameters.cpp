#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/UtilityParams.h"
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

  void affixBackwardWeightUtilityKernels(Conv2DBwdWeightOp op);
  void affixBackwardDataUtilityKernels(Conv2DBwdDataOp op);
};
} // anonymous namespace

void AffixTuningParameters::runOnOperation() {
  func::FuncOp func = getOperation();

  func.walk(
      [&](RockGemmWrapperInterface op) { affixTuningParametersImpl(op); });
  func.walk([&](Conv2DBwdDataOp op) { affixBackwardDataUtilityKernels(op); });
  func.walk(
      [&](Conv2DBwdWeightOp op) { affixBackwardWeightUtilityKernels(op); });
}

static void setUtilityKernelSizes(OpBuilder &b, Value arg, Operation *convOp,
                                  Operation *funcOp) {
  int64_t numElements = arg.getType().cast<ShapedType>().getNumElements();
  uint32_t blockSize = kUtilityKernelBlockSize;
  int64_t elemsPerThread = kUtilityKernelElemsPerThread;
  uint32_t gridSize =
      math_util::integer_divide_ceil(numElements, blockSize * elemsPerThread);

  IntegerAttr blockSizeAttr = b.getI32IntegerAttr(blockSize);
  IntegerAttr gridSizeAttr = b.getI32IntegerAttr(gridSize);

  // Tracking utility kernel block size separately.
  convOp->setAttr("utilityBlockSize", blockSizeAttr);
  convOp->setAttr("gridSize", gridSizeAttr);
  convOp->setAttr("elems_per_thread", b.getIndexAttr(elemsPerThread));

  funcOp->setAttr("grid_size", gridSizeAttr);
}

void AffixTuningParameters::affixBackwardDataUtilityKernels(
    Conv2DBwdDataOp op) {
  auto gemmIdAttr = op->template getAttrOfType<IntegerAttr>("gemm_id");

  // In case the gemm ID is -1, override grid_size and block_size for the
  // utility kernel.
  if (gemmIdAttr.getInt() < 0) {
    OpBuilder b(op.getContext());
    setUtilityKernelSizes(b, op.getInput(), op, getOperation());
  }
}

void AffixTuningParameters::affixBackwardWeightUtilityKernels(
    Conv2DBwdWeightOp op) {
  auto gemmIdAttr = op->template getAttrOfType<IntegerAttr>("gemm_id");
  assert(gemmIdAttr);
  int64_t gemmId = gemmIdAttr.getInt();

  GemmFeatures features = op.getFeatures();
  if (bitEnumContainsAll(features, GemmFeatures::mfma)) {
    OpBuilder b(op.getContext());

    GemmSize gemmSize = op.getGemmSize();

    auto gemmParams =
        op->getAttrOfType<XdlopsGemmParamsAttr>(op.getParamsAttrName());
    Optional<GemmSize> extraPadSizes = calculatePadding(
        gemmParams.getKPerBlock(), gemmParams.getMPerBlock(),
        gemmParams.getNPerBlock(), gemmSize, gemmParams.getKpack());
    if (extraPadSizes.has_value()) {
      assert(gemmId == 0 &&
             "if there is padding, only a single kernel should be generated");
    } else {
      assert((gemmId >= 0) && (gemmId < 3));
      switch (gemmId) {
      case 0:
      case 2:
        setUtilityKernelSizes(b, op.getFilter(), op, getOperation());
        break;
      case 1:
        break;
      }
    }
  }
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
  if (bitEnumContainsAll(features, GemmFeatures::mfma)) {
    PopulateParamsXDL populateParamsXDL;
    InitParamsXDL validParams;
    uint32_t blockSize = 0;
    uint32_t gridSize = 0;
    int64_t gemmKBlocks = 1;

    LogicalResult status = populateParamsXDL.obtainTuningParameters(
        op, blockSizeOverride, perfConfig, validParams, blockSize, gridSize,
        gemmKBlocks);

    if (failed(status)) {
      // Try again if allowed.
      if (fallBackNoConfig) {
        perfConfig.clear();
        status = populateParamsXDL.obtainTuningParameters(
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

    Attribute gemmParams = b.getAttr<XdlopsGemmParamsAttr>(
        validParams.gemmKPerBlock, validParams.gemmMPerBlock,
        validParams.gemmNPerBlock, validParams.gemmKPack,
        validParams.gemmMPerWave, validParams.gemmNPerWave);
    op.setGemmParamsAttr(gemmParams);

    // Set attributes on the function.
    getOperation()->setAttr("block_size", b.getI32IntegerAttr(blockSize));
    getOperation()->setAttr("grid_size", b.getI32IntegerAttr(gridSize));
  } else {
    InitParamsNonXDL validParams;
    uint32_t gridSize;

    PopulateParams populateParams;
    LogicalResult status = populateParams.obtainTuningParameters(
        op, blockSizeOverride, perfConfig, validParams, gridSize);

    if (failed(status)) {
      signalPassFailure();
    }

    gridSize = gridSizeOverride ? gridSizeOverride : gridSize;
    op.setGridSizeAttr(b.getI32IntegerAttr(gridSize));

    // For non-XDLOPS path, do not use KPack for now.

    // kPerThread and the cuwave parameters are hardcoded, may change in a
    // different pass. Please visit
    // gridwise_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw for details

    Attribute gemmParams = b.getAttr<GeneralGemmParamsAttr>(
        validParams.blockSize, validParams.gemmKPerBlock,
        validParams.gemmMPerBlock, validParams.gemmNPerBlock,
        /*kPerThread=*/1, validParams.gemmMPerThread,
        validParams.gemmNPerThread,
        /*kpack=*/1);
    op.setGemmParamsAttr(gemmParams);

    // Set attributes on the function.
    getOperation()->setAttr("block_size",
                            b.getI32IntegerAttr(validParams.blockSize));
    getOperation()->setAttr("grid_size", b.getI32IntegerAttr(gridSize));
  }
}
