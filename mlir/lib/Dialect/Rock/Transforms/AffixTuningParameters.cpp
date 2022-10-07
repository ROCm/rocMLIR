#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/GemmContext.h"
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
  template <typename T> void affixTuningParametersImpl(T &op);

  void affixBackwardWeightUtilityKernels(Conv2DBwdWeightOp &op);
  void affixBackwardDataUtilityKernels(Conv2DBwdDataOp &op);
};
} // anonymous namespace

static ConvolutionDims obtainConvDims(Operation *op) {
  auto filterLayoutAttr = op->getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->getAttrOfType<ArrayAttr>("input_layout");
  auto outputLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("output_layout");

  // Get shape of filter tensor.
  auto filterType = op->getOperand(0).getType().template cast<MemRefType>();
  ArrayRef<int64_t> filterShape = filterType.getShape();

  // Get shape of input tensor.
  auto inputType = op->getOperand(1).getType().template cast<MemRefType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // Get shape of output tensor.
  auto outputType = op->getOperand(2).getType().template cast<MemRefType>();
  ArrayRef<int64_t> outputShape = outputType.getShape();

  int64_t y, x, ho, wo, hi, wi, k, c, n, g;
  y = x = ho = wo = hi = wi = k = c = n = g = 0;

  for (unsigned i = 0; i < filterLayoutAttr.size(); ++i) {
    auto filterAttr = filterLayoutAttr.getValue()[i].cast<StringAttr>();
    auto inputAttr = inputLayoutAttr.getValue()[i].cast<StringAttr>();
    auto outputAttr = outputLayoutAttr.getValue()[i].cast<StringAttr>();

    if (filterAttr.getValue() == "y") {
      y = filterShape[i];
    } else if (filterAttr.getValue() == "x") {
      x = filterShape[i];
    } else if (filterAttr.getValue() == "k") {
      k = filterShape[i];
    } else if (filterAttr.getValue() == "c") {
      c = filterShape[i];
    } else if (filterAttr.getValue() == "g") {
      g = filterShape[i];
    }

    if (inputAttr.getValue() == "hi") {
      hi = inputShape[i];
    } else if (inputAttr.getValue() == "wi") {
      wi = inputShape[i];
    } else if (inputAttr.getValue() == "ni") {
      n = inputShape[i];
    }

    if (outputAttr.getValue() == "ho") {
      ho = outputShape[i];
    } else if (outputAttr.getValue() == "wo") {
      wo = outputShape[i];
    }
  }

  return ConvolutionDims(y, x, ho, wo, hi, wi, k, c, n, g);
}

void AffixTuningParameters::runOnOperation() {
  func::FuncOp func = getOperation();

  func.walk([&](Conv2DOp op) { affixTuningParametersImpl(op); });
  func.walk([&](Conv2DBwdDataOp op) {
    affixTuningParametersImpl(op);
    affixBackwardDataUtilityKernels(op);
  });
  func.walk([&](Conv2DBwdWeightOp op) {
    affixTuningParametersImpl(op);
    affixBackwardWeightUtilityKernels(op);
  });
}

static void setUtilityKernelSizes(OpBuilder &b, Value arg, Operation *convOp,
                                  Operation *funcOp) {
  int64_t numElements = arg.getType().cast<MemRefType>().getNumElements();
  uint32_t blockSize = kUtilityKernelBlockSize;
  int64_t elemsPerThread = kUtilityKernelElemsPerThread;
  uint32_t gridSize =
      math_util::integer_divide_ceil(numElements, blockSize * elemsPerThread);

  IntegerAttr blockSizeAttr = b.getI32IntegerAttr(blockSize);
  IntegerAttr gridSizeAttr = b.getI32IntegerAttr(gridSize);
  convOp->setAttr("blockSize", blockSizeAttr);
  convOp->setAttr("gridSize", gridSizeAttr);
  convOp->setAttr("elems_per_thread", b.getIndexAttr(elemsPerThread));

  funcOp->setAttr("block_size", blockSizeAttr);
  funcOp->setAttr("grid_size", gridSizeAttr);
}

void AffixTuningParameters::affixBackwardDataUtilityKernels(
    Conv2DBwdDataOp &op) {
  auto gemmIdAttr = op->template getAttrOfType<IntegerAttr>("gemm_id");

  // In case the gemm ID is -1, override grid_size and block_size for the
  // utility kernel.
  if (gemmIdAttr.getInt() < 0) {
    OpBuilder b(op.getContext());
    setUtilityKernelSizes(b, op.getInput(), op, getOperation());
  }
}

void AffixTuningParameters::affixBackwardWeightUtilityKernels(
    Conv2DBwdWeightOp &op) {
  auto gemmIdAttr = op->template getAttrOfType<IntegerAttr>("gemm_id");
  assert(gemmIdAttr);
  int64_t gemmId = gemmIdAttr.getInt();

  GemmFeatures features = op.getFeatures();
  if (bitEnumContainsAll(features, GemmFeatures::mfma)) {
    OpBuilder b(op.getContext());

    ConvolutionDims convDims = obtainConvDims(op);
    GemmContext gemmSize =
        GemmContext::fromConvolution(ConvOpType::BwdWeight, convDims);

    auto gemmParams =
        op->getAttrOfType<XdlopsGemmParamsAttr>(op.getParamsAttrName());
    Optional<GemmContext> extraPadSizes = calculatePadding(
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

template <typename T>
void AffixTuningParameters::affixTuningParametersImpl(T &op) {
  OpBuilder b(op.getContext());

  ConvolutionDims dims = obtainConvDims(op);

  std::string perfConfig;
  if (auto perfConfigAttr =
          op->template getAttrOfType<StringAttr>("perf_config")) {
    perfConfig = perfConfigAttr.getValue().str();
  }
  GemmFeatures features = op.getFeatures();
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

    Type dataType = obtainConvDataType(op);

    gridSize = gridSizeOverride ? gridSizeOverride : gridSize;
    op->setAttr(op.getBlockSizeAttrName(), b.getI32IntegerAttr(blockSize));
    op->setAttr(op.getGridSizeAttrName(), b.getI32IntegerAttr(gridSize));

    // Set kblocks attribute only for backward weight convolutions.
    if (auto bwdOp = dyn_cast<Conv2DBwdWeightOp>(op.getOperation()))
      bwdOp->setAttr(bwdOp.getKBlocksAttrName(), b.getIndexAttr(gemmKBlocks));

    Attribute gemmParams = b.getAttr<XdlopsGemmParamsAttr>(
        validParams.gemmKPerBlock, validParams.gemmMPerBlock,
        validParams.gemmNPerBlock, validParams.gemmKPack,
        validParams.gemmMPerWave, validParams.gemmNPerWave);
    op->setAttr(op.getParamsAttrName(), gemmParams);

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

    op->setAttr(op.getBlockSizeAttrName(),
                b.getI32IntegerAttr(validParams.blockSize));
    op->setAttr(op.getGridSizeAttrName(), b.getI32IntegerAttr(gridSize));

    // For non-XDLOPS path, do not use KPack for now.

    // kPerThread and the cuwave parameters are hardcoded, may change in a
    // different pass. Please visit
    // gridwise_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw for details

    Attribute gemmParams = b.getAttr<GeneralGemmParamsAttr>(
        validParams.gemmKPerBlock, validParams.gemmMPerBlock,
        validParams.gemmNPerBlock,
        /*kPerThread=*/1, validParams.gemmMPerThread,
        validParams.gemmNPerThread,
        /*kpack=*/1);
    op->setAttr(op.getParamsAttrName(), gemmParams);

    // Set attributes on the function.
    getOperation()->setAttr("block_size",
                            b.getI32IntegerAttr(validParams.blockSize));
    getOperation()->setAttr("grid_size", b.getI32IntegerAttr(gridSize));
  }
}
