#include "mlir/Conversion/LinalgToRock/LinalgToRock.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_LINALGTOROCKPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

namespace {
struct LinalgToRockPass : public impl::LinalgToRockPassBase<LinalgToRockPass> {
  void runOnOperation() override;
};
} // namespace

void LinalgToRockPass::runOnOperation() {
  MLIRContext &ctx = getContext();
  func::FuncOp func = getOperation();

  ConversionTarget bodyConversionTarget(ctx);
  TypeConverter converter;
  RewritePatternSet bodyPatterns(&ctx);
  rock::populateLinalgToRockConversionPattern(bodyPatterns, &ctx);
  bodyConversionTarget.addLegalDialect<
      arith::ArithDialect, tensor::TensorDialect, rock::RockDialect, bufferization::BufferizationDialect>();
  bodyConversionTarget.addIllegalDialect<linalg::LinalgDialect>();
  if (failed(applyPartialConversion(func, bodyConversionTarget,
                                    std::move(bodyPatterns)))) {
    return signalPassFailure();
  }
}
