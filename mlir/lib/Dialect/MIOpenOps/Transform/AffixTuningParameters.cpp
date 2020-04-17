#include "mlir/Dialect/MIOpenOps/MIOpenOps.h"
#include "mlir/Dialect/MIOpenOps/Passes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct AffixTuningParameters : public FunctionPass<AffixTuningParameters> {
  void runOnFunction() override;
};
} // anonymous namespace

void AffixTuningParameters::runOnFunction() {
  FuncOp func = getFunction();

  func.walk([&](miopen::GridwiseGemmOp op) {
    OpBuilder b(op.getContext());

    // TBD. Compute tuning parameters from actual logic.
    op.setAttr("block_size", b.getI32IntegerAttr(256));

    op.setAttr("m_per_block", b.getI32IntegerAttr(128));
    op.setAttr("n_per_block", b.getI32IntegerAttr(128));
    op.setAttr("k_per_block", b.getI32IntegerAttr(8));

    op.setAttr("m_per_thread", b.getI32IntegerAttr(4));
    op.setAttr("n_per_thread", b.getI32IntegerAttr(4));
    op.setAttr("k_per_thread", b.getI32IntegerAttr(4));

    op.setAttr("m_level0_cluster", b.getI32IntegerAttr(4));
    op.setAttr("n_level0_cluster", b.getI32IntegerAttr(4));
    op.setAttr("m_level1_cluster", b.getI32IntegerAttr(4));
    op.setAttr("n_level1_cluster", b.getI32IntegerAttr(4));

    op.setAttr("matrix_a_source_vector_read_dim", b.getI32IntegerAttr(0));
    op.setAttr("matrix_a_source_data_per_read", b.getI32IntegerAttr(4));
    op.setAttr("matrix_a_dest_data_per_write_dim_m", b.getI32IntegerAttr(4));

    op.setAttr("matrix_b_source_vector_read_dim", b.getI32IntegerAttr(1));
    op.setAttr("matrix_b_source_data_per_read", b.getI32IntegerAttr(4));
    op.setAttr("matrix_b_dest_data_per_write_dim_n", b.getI32IntegerAttr(4));

    op.setAttr("matrix_c_source_dest_vector_read_write_dim", b.getI32IntegerAttr(3));
    op.setAttr("matrix_c_dest_data_per_write", b.getI32IntegerAttr(1));
  });
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::miopen::createAffixTuningParametersPass() {
  return std::make_unique<AffixTuningParameters>();
}

static PassRegistration<AffixTuningParameters>
  pass("miopen-affix-params", "Affix tuning parameters to miopen.gridwise_gemm operations");

