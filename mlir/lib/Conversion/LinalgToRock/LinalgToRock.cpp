#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Conversion/LinalgToRock/LinalgToRock.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {
// FIXME: make this a template?
struct MatmulConverter final
    : public OpConversionPattern<linalg::BatchMatmulOp> {
  using OpConversionPattern<linalg::BatchMatmulOp>::OpConversionPattern;
  using OpConversionPattern<linalg::BatchMatmulOp>::getTypeConverter;
  using OpAdaptor =
      typename OpConversionPattern<linalg::BatchMatmulOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(linalg::BatchMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
MatmulConverter::matchAndRewrite(linalg::BatchMatmulOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value a = op.getOperand(0);
  Value b = op.getOperand(1);
  Value cOriginal = op.getOperand(2);
  Value c = bufferization::AllocTensorOp::create(rewriter, op.getLoc(), cast<RankedTensorType>(cOriginal.getType()), {});

  // static GemmOp create(::mlir::OpBuilder &builder, ::mlir::Location location,
  // ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b,
  // ::mlir::Value c, /*optional*/::mlir::Value scaleA,
  // /*optional*/::mlir::Value scaleB, /*optional*/bool aTransposed,
  // /*optional*/bool bTransposed, /*optional*/bool cTransposed,
  // /*optional*/bool aScaleTransposed, /*optional*/bool bScaleTransposed,
  // /*optional*/::mlir::rock::GemmFeaturesAttr features,
  // ::mlir::rock::StoreMethod storeMethod, /*optional*/::mlir::IntegerAttr
  // derivedBlockSize, /*optional*/::mlir::IntegerAttr gridSize,
  // /*optional*/::mlir::rock::RockTuningParamAttrInterface params);
  rock::StoreMethodAttr method =
      rewriter.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::Set);
  auto result = rock::GemmOp::create(
      rewriter, loc, c.getType(), a, b, c, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, method, nullptr, nullptr, nullptr);
  rewriter.replaceOp(op, result);
  return success();
}

void mlir::rock::populateLinalgToRockConversionPattern(
    RewritePatternSet &pattern, MLIRContext *context) {
  pattern.add<MatmulConverter>(context);
}
