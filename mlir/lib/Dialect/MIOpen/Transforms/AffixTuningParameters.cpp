#include "PassDetail.h"

#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/MIOpenCPP.h"

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

  // In the original C++ implementation, template arguments used by GridwiseGemmOp are:
  // template <index_t GridSize,                                 - NOT USED
  //           index_t BlockSize,                                - block_size attribute
  //           typename Float,                                   - element type of input/output/filter memrefs
  //           typename AccFloat,                                - NOT modeled yet. Will need to add it when working on bf16/fp16 data types.
  //           typename AGlobalDesc,                             - filter memref
  //           typename BGlobalDesc,                             - input memref
  //           typename CGlobalDesc,                             - output memref
  //           InMemoryDataOperation CGlobalMemoryDataOperation, - NOT USED
  //           index_t MPerBlock,                                - m_per_block attribute
  //           index_t NPerBlock,                                - n_per_block attribute
  //           index_t KPerBlock,                                - k_per_block attribute
  //           index_t MPerThread,                               - m_per_thread attribute
  //           index_t NPerThread,                               - n_per_thread attribute
  //           index_t KPerThread,                               - k_per_thread attribute
  //           index_t MLevel0Cluster,                           - m_level0_cluster attribute
  //           index_t NLevel0Cluster,                           - n_level0_cluster attribute
  //           index_t MLevel1Cluster,                           - m_level1_cluster attribute
  //           index_t NLevel1Cluster,                           - n_level1_cluster attribute
  //           index_t ThreadGemmAThreadCopySrcDataPerRead_M,    - m_per_thread attribute
  //           index_t ThreadGemmBThreadCopySrcDataPerRead_N,    - n_per_thread attribute
  //           typename ABlockCopyThreadSliceLengths_K_M,        - 
  //                                                               - GemmABlockCopyThreadSliceLengths_GemmK_GemmM in GridwiseConvolutionImplicitGemm_v4r4_nchw_kcyx_nkhw
  //                                                               - Sequence<GemmABlockCopyThreadSliceLengths_GemmK, GemmABlockCopyThreadSliceLengths_GemmM>
  //
  //                                                                 - GemmABlockCopyThreadSliceLengths_GemmK =
  //                                                                   k_per_block / GemmABlockCopyClusterLengths_GemmK
  //
  //                                                                 - GemmABlockCopyThreadSliceLengths_GemmM =
  //                                                                   m_per_block / GemmABlockCopyClusterLengths_GemmM
  //
  //           typename ABlockCopyThreadClusterLengths_K_M,      -
  //                                                               - GemmABlockCopyThreadClusterLengths_GemmK_GemmM in GridwiseConvolutionImplicitGemm_v4r4_nchw_kcyx_nkhw
  //                                                               - Sequence<GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM>
  //
  //                                                                 - GemmABlockCopyClusterLengths_GemmK =
  //                                                                   k_per_block / matrix_a_source_data_per_read
  //
  //                                                                 - GemmABlockCopyClusterLengths_GemmM =
  //                                                                   m_per_block / ((m_per_block * k_per_lbock / block_size) / matrix_a_source_data_per_read)
  //
  //           typename ABlockCopyThreadClusterArrangeOrder,     - Sequence<1, 0>
  //           typename ABlockCopySrcAccessOrder,                - Sequence<1, 0>
  //           index_t ABlockCopySrcVectorReadDim,               - matrix_a_source_vector_read_dim attribute
  //           index_t ABlockCopySrcDataPerRead,                 - matrix_a_source_data_per_read attribute
  //           index_t ABlockCopyDstDataPerWrite_M,              - matrix_a_dest_data_per_write_dim_m attribute
  //           typename BBlockCopyThreadSliceLengths_K_N,        - 
  //                                                               - GemmBBlockCopyThreadSliceLengths_GemmK_GemmN in GridwiseConvolutionImplicitGemm_v4r4_nchw_kcyx_nkhw
  //                                                               - Sequence<GemmBBlockCopyThreadSliceLengths_GemmK, GemmBBlockCopyThreadSliceLengths_GemmN>
  //                                                                 - GemmBBlockCopyThreadSliceLengths_GemmK =
  //                                                                   k_per_block / GemmBBlockCopyClusterLengths_GemmK
  //
  //                                                                 - GemmBBlockCopyThreadSliceLengths_GemmN =
  //                                                                   n_per_block / GemmBBlockCopyClusterLengths_GemmN
  //
  //           typename BBlockCopyThreadClusterLengths_K_N,      -
  //                                                               - GemmBBlockCopyThreadClusterLengths_GemmK_GemmN in GridwiseConvolutionImplicitGemm_v4r4_nchw_kcyx_nkhw
  //                                                               - Sequence<GemmBBlockCopyClusterLengths_GemmK, GemmBBlockCopyClusterLengths_GemmN>
  //
  //                                                                 - GemmBBlockCopyClusterLengths_GemmK =
  //                                                                   k_per_block / ((n_per_block * k_per_lbock / block_size) / matrix_b_source_data_per_read)
  //
  //                                                                 - GemmBBlockCopyClusterLengths_GemmN =
  //                                                                   n_per_block / matrix_b_source_data_per_read
  //
  //           typename BBlockCopyThreadClusterArrangeOrder,     - Sequence<0, 1>
  //           typename BBlockCopySrcAccessOrder,                - Sequence<0, 1>
  //           index_t BBlockCopySrcVectorReadDim,               - matrix_b_source_dest_vector_read_write_dim attribute
  //           index_t BBlockCopySrcDataPerRead,                 - matrix_b_source_data_per_read attribute
  //           index_t BBlockCopyDstDataPerWrite_N,              - matrix_b_dest_data_per_write_dim_n attribute
  //           typename CThreadCopySrcDstAccessOrder,            - Sequence<0, 1, 2, 3>
  //           index_t CThreadCopySrcDstVectorReadWriteDim,      - matrix_c_source_dest_vector_read_write_dim attribute
  //           index_t CThreadCopyDstDataPerWrite>               - matrix_c_dest_data_per_write attribute

  func.walk([&](miopen::GridwiseGemmOp op) { affixTuningParametersImpl(op); });

  func.walk(
      [&](miopen::GridwiseGemmV2Op op) { affixTuningParametersImpl(op); });
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
