#include "PassDetail.h"

#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/Tuning/ConvContext.h"
#include "mlir/Dialect/MIOpen/Tuning/GemmContext.h"
#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/MIOpen/Tuning/UtilityParams.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"
#include "mlir/Dialect/MIOpen/utility/math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::miopen;

namespace {
struct AffixTuningParameters
    : public MIOpenOpsAffixTuningParametersPassBase<AffixTuningParameters> {
public:
  AffixTuningParameters(int64_t blockSizeOverride, int64_t gridSizeOverride,
                        bool fallBackNoConfig)
      : blockSizeOverride(blockSizeOverride),
        gridSizeOverride(gridSizeOverride), fallBackNoConfig(fallBackNoConfig) {
  }
  void runOnOperation() override;

private:
  // Block size can be set in two ways:
  // * Through the MLIR lowering pass:
  //   At this case, blockSizeOverride will be initialized to zero. Then
  //   the affix tuning parameters pass will decide on a block size.
  //   Finally, block size will be hinted back to mlir-miopen-driver.
  // * Through cmd option "block_size":
  //   At this case, mlir-miopen-driver assigns a blockSizeOverride. As
  //   a result, affix tuning parameters pass should make its decisions
  //   to generate tuning parameters based on this blockSizeOverride.
  //   This guarantess that affix tuning parameters pass generate
  //   coherent tuning parameters with the pre-set block size.
  int64_t blockSizeOverride;
  int64_t gridSizeOverride;
  bool fallBackNoConfig;

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
  int64_t blockSize = kUtilityKernelBlockSize;
  int64_t elemsPerThread = kUtilityKernelElemsPerThread;
  int64_t gridSize =
      math_util::integer_divide_ceil(numElements, blockSize * elemsPerThread);
  SmallVector<Operation *, 2> ops = {convOp, funcOp};
  for (Operation *op : ops) {
    op->setAttr("grid_size", b.getI32IntegerAttr(gridSize));
    op->setAttr("block_size", b.getI32IntegerAttr(blockSize));
    op->setAttr("elems_per_thread", b.getI32IntegerAttr(elemsPerThread));
  }
}

void AffixTuningParameters::affixBackwardDataUtilityKernels(
    Conv2DBwdDataOp &op) {
  auto gemmIdAttr = op->template getAttrOfType<IntegerAttr>("gemm_id");

  // In case the gemm ID is -1, override grid_size and block_size for the
  // utility kernel.
  if (gemmIdAttr.getInt() < 0) {
    OpBuilder b(op.getContext());
    setUtilityKernelSizes(b, op.input(), op, getOperation());
  }
}

void AffixTuningParameters::affixBackwardWeightUtilityKernels(
    Conv2DBwdWeightOp &op) {
  auto gemmIdAttr = op->template getAttrOfType<IntegerAttr>("gemm_id");
  assert(gemmIdAttr);
  int64_t gemmId = gemmIdAttr.getInt();

  auto xdlopsV2Attr = op->template getAttrOfType<BoolAttr>("xdlopsV2");
  if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true) {
    OpBuilder b(op.getContext());

    ConvolutionDims convDims = obtainConvDims(op);
    GemmContext gemmSize =
        GemmContext::fromConvolution(ConvOpType::BwdWeight, convDims);

    PopulateParamsXDL populateParamsXDL;
    Optional<GemmContext> extraPadSizes =
        calculatePaddingKernelSize(gemmSize, obtainConvDirection(op),
                                   obtainConvDataType(op), populateParamsXDL);

    // For padding cases, gemmId must be 0.
    if (extraPadSizes.hasValue()) {
      assert(gemmId == 0);
    } else {
      assert((gemmId >= 0) && (gemmId < 3));
      switch (gemmId) {
      case 0:
      case 2:
        setUtilityKernelSizes(b, op.filter(), op, getOperation());
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
  ConvOpType opType = obtainConvDirection(op);
  GemmContext gemmSize = GemmContext::fromConvolution(opType, dims);

  std::string perfConfig;
  if (auto perfConfigAttr =
          op->template getAttrOfType<StringAttr>("perf_config")) {
    perfConfig = perfConfigAttr.getValue().str();
  }
  auto xdlopsV2Attr = op->template getAttrOfType<BoolAttr>("xdlopsV2");
  if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true) {
    PopulateParamsXDL populateParamsXDL;
    InitParamsXDL validParams;
    DerivedParams gemmADerivedParam;
    DerivedParams gemmBDerivedParam;
    DerivedOutParams gemmCDerivedParam;
    int64_t blockSize = 0;
    int64_t gridSize = 0;
    int64_t gemmKBlocks = 1;

    LogicalResult status = populateParamsXDL.obtainTuningParameters(
        op, blockSizeOverride, perfConfig, validParams, gemmADerivedParam,
        gemmBDerivedParam, gemmCDerivedParam, blockSize, gridSize, gemmKBlocks);

    if (failed(status)) {
      // Try again if allowed.
      if (fallBackNoConfig) {
        perfConfig.clear();
        status = populateParamsXDL.obtainTuningParameters(
            op, blockSizeOverride, perfConfig, validParams, gemmADerivedParam,
            gemmBDerivedParam, gemmCDerivedParam, blockSize, gridSize,
            gemmKBlocks);
      }
      if (failed(status))
        signalPassFailure();
    }

    ConvOpType dir = obtainConvDirection(op);
    Type dataType = obtainConvDataType(op);

    // Disable kpack in case we need padding kernel.
    Optional<GemmContext> gemmExtraPad =
        calculatePaddingKernelSize(gemmSize, dir, dataType, populateParamsXDL);
    if (gemmExtraPad.hasValue()) {
      validParams.gemmKPack = 1;
    }

    op->setAttr(op.blockSizeAttrName(), b.getIndexAttr(blockSize));
    op->setAttr(op.gridSizeAttrName(), b.getIndexAttr(gridSize));

    // Set kblocks attribute only for backward weight convolutions.
    if (auto bwdOp = dyn_cast<Conv2DBwdWeightOp>(op.getOperation()))
      bwdOp->setAttr(bwdOp.kBlocksAttrName(), b.getIndexAttr(gemmKBlocks));

    Attribute gemmParams = XdlopsGemmParamsAttr::get(
        op.getContext(), validParams.gemmKPerBlock, validParams.gemmMPerBlock,
        validParams.gemmNPerBlock, validParams.gemmKPack,
        validParams.gemmMPerWave, validParams.gemmNPerWave);
    op->setAttr(op.paramsAttrName(), gemmParams);

    // Set attributes on the function.
    getOperation()->setAttr("block_size", b.getI32IntegerAttr(blockSize));
    getOperation()->setAttr(
        "grid_size",
        b.getI32IntegerAttr(gridSizeOverride ? gridSizeOverride : gridSize));

    // Derived parameters for gemmA.
    // All this goes away after new-style vectorization is implemented.
    op->setAttr("matrix_a_source_data_per_read",
                b.getI32IntegerAttr(gemmADerivedParam.srcDataPerRead));
    op->setAttr("matrix_a_source_vector_read_dim",
                b.getI32IntegerAttr(gemmADerivedParam.srcVectorReadDim));

    // Derived parameters for gemmB.
    op->setAttr("matrix_b_source_data_per_read",
                b.getI32IntegerAttr(gemmBDerivedParam.srcDataPerRead));
    op->setAttr("matrix_b_source_vector_read_dim",
                b.getI32IntegerAttr(gemmBDerivedParam.srcVectorReadDim));

    op->setAttr("matrix_c_data_per_copy",
                b.getI32IntegerAttr(gemmCDerivedParam.dataPerCopy));
    op->setAttr("matrix_c_source_vector_read_dim",
                b.getI32IntegerAttr(gemmCDerivedParam.gemmVectorDim));
    op->setAttr("matrix_c_dest_vector_write_dim",
                b.getI32IntegerAttr(gemmCDerivedParam.destVectorDim));
  } else {
    InitParamsNonXDL validParams;
    DerivedParams gemmADerivedParam;
    DerivedParams gemmBDerivedParam;
    DerivedBlockGemmParams blockGemmDerivedParam;
    DerivedOutParams gemmCDerivedParam;
    int64_t gridSize;

    PopulateParams populateParams;
    LogicalResult status = populateParams.obtainTuningParameters(
        op, blockSizeOverride, perfConfig, validParams, gemmADerivedParam,
        gemmBDerivedParam, blockGemmDerivedParam, gemmCDerivedParam, gridSize);

    if (failed(status)) {
      signalPassFailure();
    }

    gridSize = gridSizeOverride ? gridSizeOverride : gridSize;

    op->setAttr(op.blockSizeAttrName(), b.getIndexAttr(validParams.blockSize));
    op->setAttr(op.gridSizeAttrName(), b.getIndexAttr(gridSize));

    // For non-XDLOPS path, do not use KPack for now.

    // kPerThread and the cuwave parameters are hardcoded, may change in a
    // different pass. Please visit
    // gridwise_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw for details

    Attribute gemmParams = GeneralGemmParamsAttr::get(
        op->getContext(), validParams.gemmKPerBlock, validParams.gemmMPerBlock,
        validParams.gemmNPerBlock,
        /*kPerThread=*/1, validParams.gemmMPerThread,
        validParams.gemmNPerThread,
        /*kpack=*/1, blockGemmDerivedParam.gemmMThreadsPerCuwave,
        blockGemmDerivedParam.gemmNThreadsPerCuwave,
        blockGemmDerivedParam.gemmMCuwavesPerBlock,
        blockGemmDerivedParam.gemmNCuwavesPerBlock);
    op->setAttr(op.paramsAttrName(), gemmParams);

    // Set attributes on the function.
    getOperation()->setAttr("block_size",
                            b.getI32IntegerAttr(validParams.blockSize));
    getOperation()->setAttr("grid_size", b.getI32IntegerAttr(gridSize));

    // Derived parameters for gemmA.
    // Will be dead with new vectorization scheme.
    op->setAttr("matrix_a_source_data_per_read",
                b.getI32IntegerAttr(gemmADerivedParam.srcDataPerRead));
    op->setAttr("matrix_a_source_vector_read_dim",
                b.getI32IntegerAttr(gemmADerivedParam.srcVectorReadDim));

    // Derived parameters for gemmB.
    op->setAttr("matrix_b_source_data_per_read",
                b.getI32IntegerAttr(gemmBDerivedParam.srcDataPerRead));
    op->setAttr("matrix_b_source_vector_read_dim",
                b.getI32IntegerAttr(gemmBDerivedParam.srcVectorReadDim));

    op->setAttr("matrix_c_data_per_copy",
                b.getI32IntegerAttr(gemmCDerivedParam.dataPerCopy));
    op->setAttr("matrix_c_source_vector_read_dim",
                b.getI32IntegerAttr(gemmCDerivedParam.gemmVectorDim));
    op->setAttr("matrix_c_dest_vector_write_dim",
                b.getI32IntegerAttr(gemmCDerivedParam.destVectorDim));
  }
}

std::unique_ptr<Pass>
mlir::miopen::createAffixTuningParametersPass(int64_t blockSizeOverride,
                                              int64_t gridSizeOverride,
                                              bool fallBackNoConfig) {
  return std::make_unique<AffixTuningParameters>(
      blockSizeOverride, gridSizeOverride, fallBackNoConfig);
}
