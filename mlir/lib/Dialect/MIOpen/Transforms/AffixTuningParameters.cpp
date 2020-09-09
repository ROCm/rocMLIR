#include "PassDetail.h"

#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/gridwise_gemm_params.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/MIOpenCPP.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct AffixTuningParameters : public MIOpenOpsAffixTuningParametersPassBase<AffixTuningParameters> {
public:
  AffixTuningParameters(int64_t blockSizeOverride,
                        LaunchDimensionCallback launchDimCallback)
      : launchDimCallback(launchDimCallback),
        blockSizeOverride(blockSizeOverride) {}
  void runOnFunction() override;

private:
  LaunchDimensionCallback launchDimCallback;
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

  // Actual implementation.
  template<typename T>
  void affixTuningParametersImpl(T &op);
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

  func.walk([&](miopen::GridwiseGemmOp op) {
    affixTuningParametersImpl(op);
  });

  func.walk([&](miopen::GridwiseGemmV2Op op) {
    affixTuningParametersImpl(op);
  });
}


template<typename T>
void AffixTuningParameters::affixTuningParametersImpl(T &op) {
  OpBuilder b(op.getContext());

  ConvolutionContext convContext = populateConvContext(op);
  InitParamsNonXDL validParams{0, 0, 0, 0, 0, blockSizeOverride};
  GemmSize gemmSize;
  DerivedParams gemmADerivedParam;
  DerivedParams gemmBDerivedParam;
  DerivedBlockGemmParams blockGemmDerivedParam;
  int64_t gemmCDstPerWrite;
  int64_t gridSize;

  PopulateParams populateParams;
  LogicalResult status = populateParams.paramsFromCtx(
      convContext, validParams, gemmSize, gemmADerivedParam,
      gemmBDerivedParam, blockGemmDerivedParam, gemmCDstPerWrite, gridSize);
  if (failed(status)) {
    signalPassFailure();
  }

  // XXX. Populate default tuning parameters for XDLOPS.
  auto xdlopsAttr = op.template getAttrOfType<BoolAttr>("xdlops");
  auto xdlopsV2Attr = op.template getAttrOfType<BoolAttr>("xdlopsV2");
  if ((xdlopsAttr && xdlopsAttr.getValue() == true) ||
      (xdlopsV2Attr && xdlopsV2Attr.getValue() == true)) {
    validParams.gemmMPerBlock = 256;
    validParams.gemmNPerBlock = 128;
    validParams.gemmKPerBlock = 16;
    validParams.gemmMPerThread = 128;
    validParams.gemmNPerThread = 64;
    validParams.blockSize = 256;

    // XXX. fix gridSize.
    // need to use (M/MPerBlock)*(N/NPerBlock).
    gridSize = 784;
  }

  if (launchDimCallback) {
    launchDimCallback(validParams.blockSize, gridSize);
  }

  // Tunable parameters.
  op.setAttr("block_size", b.getI32IntegerAttr(validParams.blockSize));
  op.setAttr("m_per_block", b.getI32IntegerAttr(validParams.gemmMPerBlock));
  op.setAttr("n_per_block", b.getI32IntegerAttr(validParams.gemmNPerBlock));
  op.setAttr("k_per_block", b.getI32IntegerAttr(validParams.gemmKPerBlock));
  op.setAttr("m_per_thread", b.getI32IntegerAttr(validParams.gemmMPerThread));
  op.setAttr("n_per_thread", b.getI32IntegerAttr(validParams.gemmNPerThread));

  // Derived parameters for gemmA.
  op.setAttr("matrix_a_source_data_per_read",
             b.getI32IntegerAttr(gemmADerivedParam.srcDataPerRead));
  op.setAttr("matrix_a_dest_data_per_write_dim_m",
             b.getI32IntegerAttr(gemmADerivedParam.dstDataPerWrite));

  // Derived parameters for gemmB.
  op.setAttr("matrix_b_source_data_per_read",
             b.getI32IntegerAttr(gemmBDerivedParam.srcDataPerRead));
  op.setAttr("matrix_b_dest_data_per_write_dim_n",
             b.getI32IntegerAttr(gemmBDerivedParam.dstDataPerWrite));

  // Derived parameters for gemmC.
  // TODO: Pending fix from
  // https://github.com/whchung/llvm-project/pull/26/files#r444968168
  // op.setAttr("matrix_c_dest_data_per_write",
  //           b.getI32IntegerAttr(gemmCDstPerWrite));
  op.setAttr("matrix_c_dest_data_per_write", b.getI32IntegerAttr(1));

  // Hard coded parameters, will change in a different pass. Please visit
  // gridwise_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw for details
  op.setAttr("k_per_thread", b.getI32IntegerAttr(1));
  op.setAttr("m_level0_cluster", b.getI32IntegerAttr(4));
  op.setAttr("n_level0_cluster", b.getI32IntegerAttr(4));
  op.setAttr("m_level1_cluster", b.getI32IntegerAttr(4));
  op.setAttr("n_level1_cluster", b.getI32IntegerAttr(4));
  op.setAttr("matrix_a_source_vector_read_dim", b.getI32IntegerAttr(0));
  op.setAttr("matrix_b_source_vector_read_dim", b.getI32IntegerAttr(1));
  op.setAttr("matrix_c_source_dest_vector_read_write_dim",
             b.getI32IntegerAttr(3)); 
}

std::unique_ptr<Pass> mlir::miopen::createAffixTuningParametersPass(
    int64_t blockSizeOverride, LaunchDimensionCallback launchDimCallback) {
  return std::make_unique<AffixTuningParameters>(blockSizeOverride,
                                                 launchDimCallback);
}
