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

  func.walk([&](miopen::GridwiseGemmOp op) { affixTuningParametersImpl(op); });

  func.walk(
      [&](miopen::GridwiseGemmV2Op op) { affixTuningParametersImpl(op); });

  func.walk([&](miopen::Conv2DOp op) { affixTuningParametersImpl(op); });
  func.walk([&](miopen::Conv2DBwdDataOp op) { affixTuningParametersImpl(op); });
  func.walk([&](miopen::Conv2DBwdWeightOp op) { affixTuningParametersImpl(op); });
}

template <typename T>
void AffixTuningParameters::affixTuningParametersImpl(T &op) {
  OpBuilder b(op.getContext());

  ConvolutionContext convContext = populateConvContext(op);

  auto xdlopsAttr = op->template getAttrOfType<BoolAttr>("xdlops");
  auto xdlopsV2Attr = op->template getAttrOfType<BoolAttr>("xdlopsV2");
  if ((xdlopsAttr && xdlopsAttr.getValue() == true) ||
      (xdlopsV2Attr && xdlopsV2Attr.getValue() == true)) {

    PopulateParamsXDL populateParamsXDL;
    InitParamsXDL validParams;
    DerivedParams gemmADerivedParam;
    DerivedParams gemmBDerivedParam;
    int64_t blockSize = 0;
    int64_t gridSize = 0;

    LogicalResult status = populateParamsXDL.paramsFromCtx(
        convContext, blockSizeOverride, perfConfig, validParams,
        gemmADerivedParam, gemmBDerivedParam, blockSize, gridSize);

    if (failed(status)) {
      signalPassFailure();
    }

    op->setAttr("m_per_wave", b.getI32IntegerAttr(validParams.gemmMPerWave));
    op->setAttr("n_per_wave", b.getI32IntegerAttr(validParams.gemmNPerWave));
    op->setAttr("block_size", b.getI32IntegerAttr(blockSize));

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

  } else {
    InitParamsNonXDL validParams;
    DerivedParams gemmADerivedParam;
    DerivedParams gemmBDerivedParam;
    DerivedBlockGemmParams blockGemmDerivedParam;
    int64_t gemmCDstPerWrite;
    int64_t gridSize;

    PopulateParams populateParams;
    LogicalResult status = populateParams.paramsFromCtx(
        convContext, blockSizeOverride, perfConfig, validParams,
        gemmADerivedParam, gemmBDerivedParam, blockGemmDerivedParam,
        gemmCDstPerWrite, gridSize);

    if (failed(status)) {
      signalPassFailure();
    }

    op->setAttr("m_per_thread",
                b.getI32IntegerAttr(validParams.gemmMPerThread));
    op->setAttr("n_per_thread",
                b.getI32IntegerAttr(validParams.gemmNPerThread));
    op->setAttr("block_size", b.getI32IntegerAttr(validParams.blockSize));

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
  }

  // Derived parameters for gemmC.
  // TODO: Pending fix from
  // https://github.com/whchung/llvm-project/pull/26/files#r444968168
  // op->setAttr("matrix_c_dest_data_per_write",
  //           b.getI32IntegerAttr(gemmCDstPerWrite));
  op->setAttr("matrix_c_dest_data_per_write", b.getI32IntegerAttr(1));
  op->setAttr("matrix_c_source_dest_vector_read_write_dim",
              b.getI32IntegerAttr(4));
}

std::unique_ptr<Pass>
mlir::miopen::createAffixTuningParametersPass(int64_t blockSizeOverride,
                                              int64_t gridSizeOverride,
                                              std::string perfConfig) {
  return std::make_unique<AffixTuningParameters>(blockSizeOverride,
                                                 gridSizeOverride, perfConfig);
}
