#include "PassDetail.h"

#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct AffixTuningParameters : public MIOpenOpsAffixTuningParametersPassBase<AffixTuningParameters> {
public:
  AffixTuningParameters(int64_t blockSizeOverride, int64_t gridSizeOverride)
      : blockSizeOverride(blockSizeOverride),
        gridSizeOverride(gridSizeOverride) {}
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

  // Actual implementation.
  template <typename T> void affixTuningParametersImpl(T &op);

  void affixBackwardWeightUtilityKernels(miopen::Conv2DBwdWeightOp &op);

  template <typename T>
  std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
             int64_t, int64_t>
  fetchDimensions(T &op);
};
} // anonymous namespace

void AffixTuningParameters::runOnOperation() {
  FuncOp func = getOperation();

  func.walk([&](miopen::Conv2DOp op) { affixTuningParametersImpl(op); });
  func.walk([&](miopen::Conv2DBwdDataOp op) { affixTuningParametersImpl(op); });
  func.walk([&](miopen::Conv2DBwdWeightOp op) {
    affixTuningParametersImpl(op);
    affixBackwardWeightUtilityKernels(op);
  });
}

template <typename T>
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
           int64_t, int64_t>
AffixTuningParameters::fetchDimensions(T &op) {
  auto filterLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->template getAttrOfType<ArrayAttr>("input_layout");
  auto outputLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("output_layout");

  // Get shape of filter tensor.
  auto filterType = op.filter().getType().template cast<MemRefType>();
  auto filterShape = filterType.getShape();

  // Get shape of input tensor.
  auto inputType = op.input().getType().template cast<MemRefType>();
  auto inputShape = inputType.getShape();

  // Get shape of output tensor.
  auto outputType = op.output().getType().template cast<MemRefType>();
  auto outputShape = outputType.getShape();

  // get y, x, ho, wo, hi, wi, k, c, n
  int64_t y, x, ho, wo, hi, wi, k, c, n;
  y = x = ho = wo = hi = wi = k = c = n = 0;

  for (unsigned i = 0; i < filterLayoutAttr.size(); ++i) {
    auto filterAttr =
        filterLayoutAttr.getValue()[i].template cast<StringAttr>();
    auto inputAttr = inputLayoutAttr.getValue()[i].template cast<StringAttr>();
    auto outputAttr =
        outputLayoutAttr.getValue()[i].template cast<StringAttr>();

    if (filterAttr.getValue() == "y") {
      y = filterShape[i];
    } else if (filterAttr.getValue() == "x") {
      x = filterShape[i];
    } else if (filterAttr.getValue() == "k") {
      k = filterShape[i];
    } else if (filterAttr.getValue() == "c") {
      c = filterShape[i];
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

  return std::make_tuple(y, x, ho, wo, hi, wi, k, c, n);
}

void AffixTuningParameters::affixBackwardWeightUtilityKernels(
    miopen::Conv2DBwdWeightOp &op) {
  auto gemmIdAttr = op->template getAttrOfType<IntegerAttr>("gemm_id");
  assert(gemmIdAttr);
  int64_t gemmId = gemmIdAttr.getInt();

  auto xdlopsV2Attr = op->template getAttrOfType<BoolAttr>("xdlopsV2");
  if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true) {
    OpBuilder b(op.getContext());

    ConvolutionContext convContext = populateConvContext(op);

    // get y, x, ho, wo, hi, wi, k, c, n
    int64_t y, x, ho, wo, hi, wi, k, c, n;
    std::tie(y, x, ho, wo, hi, wi, k, c, n) = fetchDimensions(op);

    int64_t gemmMSize, gemmNSize, gemmKSize;
    gemmMSize = k;
    gemmKSize = n * ho * wo;
    gemmNSize = c * y * x;

    int64_t gemmMExtra, gemmNExtra, gemmKExtra;
    gemmMExtra = gemmNExtra = gemmKExtra = 0;

    // isOriginalKernelSupport is not used.
    // Only needExtraPad is used.
    bool isOriginalKernelSupport = true;
    bool needExtraPad = false;
    PopulateParamsXDL populateParamsXDL;
    std::tie(isOriginalKernelSupport, needExtraPad, gemmMExtra, gemmNExtra,
             gemmKExtra) =
        calculatePaddingKernelSize(
            gemmMSize, gemmNSize, gemmKSize, convContext.getOpType(),
            convContext.getDataType(), populateParamsXDL);

    // For padding cases, gemmId must be 0.
    if (needExtraPad == true) {
      assert(gemmId == 0);
    } else {
      assert((gemmId >= 0) && (gemmId < 3));
      switch (gemmId) {
      case 0:
      case 2:
        // Override grid_size and block_size be 1 for utility kernels.
        // FIXME. Use better sizes for speedups.
        op->setAttr("grid_size", b.getI32IntegerAttr(1));
        op->setAttr("block_size", b.getI32IntegerAttr(1));
        // Set attributes on the function.
        getOperation()->setAttr("block_size", b.getI32IntegerAttr(1));
        getOperation()->setAttr("grid_size", b.getI32IntegerAttr(1));
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

  ConvolutionContext convContext = populateConvContext(op);

  // get y, x, ho, wo, hi, wi, k, c, n
  int64_t y, x, ho, wo, hi, wi, k, c, n;
  std::tie(y, x, ho, wo, hi, wi, k, c, n) = fetchDimensions(op);

  int64_t gemmMSize, gemmNSize, gemmKSize;
  // FIXME : support forward convolution only right now.
  // compute we should use extra padding kernel or not
  // c,k already / g ,so we can skip / g here
  gemmMSize = k;
  gemmKSize = c * y * x;
  gemmNSize = n * ho * wo;

  int64_t gemmMExtra, gemmNExtra, gemmKExtra;
  gemmMExtra = gemmNExtra = gemmKExtra = 0;

  // isOriginalKernelSupport is not used.
  // Only needExtraPad is used.
  bool isOriginalKernelSupport = true;
  bool needExtraPad = false;

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

    LogicalResult status = populateParamsXDL.paramsFromCtx(
        convContext, blockSizeOverride, perfConfig, validParams,
        gemmADerivedParam, gemmBDerivedParam, gemmCDerivedParam, blockSize,
        gridSize);

    if (failed(status)) {
      signalPassFailure();
    }

    op->setAttr("m_per_wave", b.getI32IntegerAttr(validParams.gemmMPerWave));
    op->setAttr("n_per_wave", b.getI32IntegerAttr(validParams.gemmNPerWave));
    op->setAttr("block_size", b.getI32IntegerAttr(blockSize));

    // Disable kpack in case we need padding kernel.
    std::tie(isOriginalKernelSupport, needExtraPad, gemmMExtra, gemmNExtra,
             gemmKExtra) =
        calculatePaddingKernelSize(
            gemmMSize, gemmNSize, gemmKSize, convContext.getOpType(),
            convContext.getDataType(), populateParamsXDL);
    if (needExtraPad) {
      validParams.gemmKPack = 1;
    }

    // Disable kpack in case we do backward convolution.
    if (convContext.opType == mlir::miopen::ConvOpType::BwdData ||
        convContext.opType == mlir::miopen::ConvOpType::BwdWeight) {
      validParams.gemmKPack = 1;
    }
    op->setAttr("kpack", b.getI32IntegerAttr(validParams.gemmKPack));

    // Set attributes on the function.
    getOperation()->setAttr("block_size", b.getI32IntegerAttr(blockSize));
    getOperation()->setAttr(
        "grid_size",
        b.getI32IntegerAttr(gridSizeOverride ? gridSizeOverride : gridSize));

    op->setAttr("m_per_block", b.getI32IntegerAttr(validParams.gemmMPerBlock));
    op->setAttr("n_per_block", b.getI32IntegerAttr(validParams.gemmNPerBlock));
    op->setAttr("k_per_block", b.getI32IntegerAttr(validParams.gemmKPerBlock));

    // Derived parameters for gemmA.
    op->setAttr("matrix_a_source_data_per_read",
                b.getI32IntegerAttr(gemmADerivedParam.srcDataPerRead));
    op->setAttr("matrix_a_dest_data_per_write_dim_m",
                b.getI32IntegerAttr(gemmADerivedParam.dstDataPerWrite));
    op->setAttr("matrix_a_source_vector_read_dim",
                b.getI32IntegerAttr(gemmADerivedParam.srcVectorReadDim));

    // Derived parameters for gemmB.
    op->setAttr("matrix_b_source_data_per_read",
                b.getI32IntegerAttr(gemmBDerivedParam.srcDataPerRead));
    op->setAttr("matrix_b_dest_data_per_write_dim_n",
                b.getI32IntegerAttr(gemmBDerivedParam.dstDataPerWrite));
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
    LogicalResult status = populateParams.paramsFromCtx(
        convContext, blockSizeOverride, perfConfig, validParams,
        gemmADerivedParam, gemmBDerivedParam, blockGemmDerivedParam,
        gemmCDerivedParam, gridSize);

    if (failed(status)) {
      signalPassFailure();
    }

    op->setAttr("m_per_thread",
                b.getI32IntegerAttr(validParams.gemmMPerThread));
    op->setAttr("n_per_thread",
                b.getI32IntegerAttr(validParams.gemmNPerThread));
    op->setAttr("block_size", b.getI32IntegerAttr(validParams.blockSize));
    // For non-XDLOPS path, do not use KPack for now.
    // XXX
    op->setAttr("kpack", b.getI32IntegerAttr(4));

    // Set attributes on the function.
    getOperation()->setAttr("block_size",
                            b.getI32IntegerAttr(validParams.blockSize));
    getOperation()->setAttr(
        "grid_size",
        b.getI32IntegerAttr(gridSizeOverride ? gridSizeOverride : gridSize));

    op->setAttr("m_per_block", b.getI32IntegerAttr(validParams.gemmMPerBlock));
    op->setAttr("n_per_block", b.getI32IntegerAttr(validParams.gemmNPerBlock));
    op->setAttr("k_per_block", b.getI32IntegerAttr(validParams.gemmKPerBlock));

    // Derived parameters for gemmA.
    op->setAttr("matrix_a_source_data_per_read",
                b.getI32IntegerAttr(gemmADerivedParam.srcDataPerRead));
    op->setAttr("matrix_a_dest_data_per_write_dim_m",
                b.getI32IntegerAttr(gemmADerivedParam.dstDataPerWrite));
    op->setAttr("matrix_a_source_vector_read_dim",
                b.getI32IntegerAttr(gemmADerivedParam.srcVectorReadDim));

    // Derived parameters for gemmB.
    op->setAttr("matrix_b_source_data_per_read",
                b.getI32IntegerAttr(gemmBDerivedParam.srcDataPerRead));
    op->setAttr("matrix_b_dest_data_per_write_dim_n",
                b.getI32IntegerAttr(gemmBDerivedParam.dstDataPerWrite));
    op->setAttr("matrix_b_source_vector_read_dim",
                b.getI32IntegerAttr(gemmBDerivedParam.srcVectorReadDim));

    // Hard coded parameters, will change in a different pass. Please visit
    // gridwise_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw for details
    op->setAttr("k_per_thread", b.getI32IntegerAttr(1));
    op->setAttr("m_level0_cluster",
                b.getI32IntegerAttr(blockGemmDerivedParam.gemmMLevel0Cluster));
    op->setAttr("n_level0_cluster",
                b.getI32IntegerAttr(blockGemmDerivedParam.gemmNLevel0Cluster));
    op->setAttr("m_level1_cluster",
                b.getI32IntegerAttr(blockGemmDerivedParam.gemmMLevel1Cluster));
    op->setAttr("n_level1_cluster",
                b.getI32IntegerAttr(blockGemmDerivedParam.gemmNLevel1Cluster));

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
                                              int64_t gridSizeOverride) {
  return std::make_unique<AffixTuningParameters>(blockSizeOverride,
                                                 gridSizeOverride);
}
