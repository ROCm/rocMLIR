#include "PassDetail.h"

#include "mlir/Dialect/MIOpen/MIOpenOps.h"
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
  AffixTuningParameters(int64_t blockSizeOverride, int64_t gridSizeOverride,
                        std::string perfConfig)
      : blockSizeOverride(blockSizeOverride),
        gridSizeOverride(gridSizeOverride), perfConfig(perfConfig) {}
  void runOnFunction() override;

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
  std::string perfConfig;

  // Actual implementation.
  template <typename T> void affixTuningParametersImpl(T &op);
};
} // anonymous namespace

void AffixTuningParameters::runOnFunction() {
  FuncOp func = getFunction();

  func.walk([&](miopen::Conv2DOp op) { affixTuningParametersImpl(op); });
  func.walk([&](miopen::Conv2DBwdDataOp op) { affixTuningParametersImpl(op); });
  func.walk(
      [&](miopen::Conv2DBwdWeightOp op) { affixTuningParametersImpl(op); });
}

template <typename T>
void AffixTuningParameters::affixTuningParametersImpl(T &op) {
  OpBuilder b(op.getContext());

  ConvolutionContext convContext = populateConvContext(op);

  auto filterLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("input_layout");
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
  llvm::DenseMap<StringRef, int> nameToDims;
  for (unsigned i = 0; i < filterLayoutAttr.size(); ++i) {
    auto filterAttr =
        filterLayoutAttr.getValue()[i].template cast<StringAttr>();
    auto inputAttr =
        inputLayoutAttr.getValue()[i].template cast<StringAttr>();
    auto outputAttr =
        outputLayoutAttr.getValue()[i].template cast<StringAttr>();

    nameToDims[filterAttr.getValue()] = i;
    nameToDims[inputAttr.getValue()] = i;
    nameToDims[outputAttr.getValue()] = i;

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

  int64_t gemmM_size, gemmN_size, gemmK_size;
  int64_t gemmMExtra, gemmNExtra, gemmKExtra;
  gemmM_size = gemmN_size = gemmK_size = 0;
  gemmMExtra = gemmNExtra = gemmKExtra = 0;
  // FIXME : support forward convolution only right now.
  // compute we should use extra padding kernel or not
  // c,k already / g ,so we can skip / g here
  gemmM_size = k;
  gemmK_size = c * y * x;
  gemmN_size = n * ho * wo;

  bool needExtraPad = false;

  auto calculatePaddingKernelSize = [&needExtraPad, gemmM_size, gemmN_size,
                                     gemmK_size, &gemmMExtra, &gemmNExtra,
                                     &gemmKExtra,
                                     &convContext](auto populateParams) {
    auto config_params = populateParams.getTuningParameters(convContext);
    unsigned numOfFailedConfigs = 0;
    for (auto &params : config_params) {
      if (gemmM_size % params.gemmMPerBlock == 0 &&
          gemmK_size % params.gemmKPerBlock == 0 &&
          gemmN_size % params.gemmNPerBlock == 0) {
        break;
      } else {
        numOfFailedConfigs++;
      }
    }

    auto extraParams = populateParams.getUniversalParameters();
    if (numOfFailedConfigs == config_params.size()) {
      needExtraPad = true;
      int gemmM_remain, gemmK_remain, gemmN_remain;

      gemmM_remain = gemmM_size % extraParams.gemmMPerBlock;
      if (gemmM_remain != 0)
        gemmMExtra = extraParams.gemmMPerBlock - gemmM_remain;

      gemmN_remain = gemmN_size % extraParams.gemmNPerBlock;
      if (gemmN_remain != 0)
        gemmNExtra = extraParams.gemmNPerBlock - gemmN_remain;

      gemmK_remain = gemmK_size % extraParams.gemmKPerBlock;
      if (gemmK_remain != 0)
        gemmKExtra = extraParams.gemmKPerBlock - gemmK_remain;

      // llvm::errs() << "gemmMExtra: " << gemmMExtra << "gemmNExtra: " <<
      // gemmNExtra << "gemmKExtra: " << gemmKExtra << "\n";
    }
  };

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
    calculatePaddingKernelSize(populateParamsXDL);
    if (needExtraPad)
      validParams.gemmKPack = 1;
    op->setAttr("kpack", b.getI32IntegerAttr(validParams.gemmKPack));

    // Set attributes on the function.
    getFunction()->setAttr("block_size", b.getI32IntegerAttr(blockSize));
    getFunction()->setAttr(
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
    op->setAttr("kpack", b.getI32IntegerAttr(1));

    // Set attributes on the function.
    getFunction()->setAttr("block_size",
                           b.getI32IntegerAttr(validParams.blockSize));
    getFunction()->setAttr(
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
                                              int64_t gridSizeOverride,
                                              std::string perfConfig) {
  return std::make_unique<AffixTuningParameters>(blockSizeOverride,
                                                 gridSizeOverride, perfConfig);
}
