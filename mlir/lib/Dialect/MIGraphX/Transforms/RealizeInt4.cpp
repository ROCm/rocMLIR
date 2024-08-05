//===- RealizeInt4.cpp ------------------------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2024 Advanced Micro Devices
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MIGraphX/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace migraphx {
#define GEN_PASS_DEF_MIGRAPHXREALIZEINT4PASS
#include "mlir/Dialect/MIGraphX/Passes.h.inc"
} // namespace migraphx
} // namespace mlir

using namespace mlir;
using namespace mlir::migraphx;

namespace {
struct MIGraphXRealizeInt4Pass
    : public migraphx::impl::MIGraphXRealizeInt4PassBase<
          MIGraphXRealizeInt4Pass> {
  void runOnOperation() override;
};

struct RewriteByteUnpackPattern : public OpConversionPattern<UnpackOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct TransposeUnpackInterchange : public OpConversionPattern<UnpackOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ReshapeUnpackInterchange : public OpConversionPattern<UnpackOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct FuncArgUnpackElimination : public OpConversionPattern<UnpackOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // end namespace

static MIXRShapedType asInt4Tensor(const MIXRShapedType byteType,
                                   int64_t axis) {
  SmallVector<int64_t, 8> sizes(byteType.getShape());
  SmallVector<int64_t, 8> strides(byteType.getStrides());
  sizes.insert(sizes.begin() + (axis + 1), 2);
  for (int64_t &stride : strides)
    stride *= 2;
  strides.insert(strides.begin() + (axis + 1), 1);
  return MIXRShapedType::get(sizes, strides,
                             IntegerType::get(byteType.getContext(), 4));
}

LogicalResult RewriteByteUnpackPattern::matchAndRewrite(
    UnpackOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  MIXRShapedType outType = op.getOut().getType();
  if (!outType.getElementType().isInteger(8))
    return failure();

  Location loc = op.getLoc();
  int64_t axis = op.getAxis();
  MIXRShapedType packedByteType = op.getIn().getType();
  MIXRShapedType actualType = asInt4Tensor(packedByteType, axis);
  Value correctedTensor =
      rewriter.create<UnpackOp>(loc, actualType, adaptor.getIn(), axis);
  MIXRShapedType unpackedType =
      actualType.cloneWith(std::nullopt, std::nullopt, rewriter.getI8Type());
  Value unpacked =
      rewriter.create<ConvertOp>(loc, unpackedType, correctedTensor,
                                 /*zeroExtend=*/rewriter.getUnitAttr());
  rewriter.replaceOpWithNewOp<ReshapeOp>(
      op, outType, unpacked, rewriter.getI64ArrayAttr(outType.getShape()));
  return success();
}

LogicalResult TransposeUnpackInterchange::matchAndRewrite(
    UnpackOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto trOp = adaptor.getIn().getDefiningOp<TransposeOp>();
  if (!trOp)
    return failure();
  size_t postTransposeAxis = op.getAxis();
  ArrayAttr permutation = trOp.getPermutation();
  size_t preTransposeAxis =
      cast<IntegerAttr>(permutation[postTransposeAxis]).getInt();

  MIXRShapedType preTrCorrectedType =
      asInt4Tensor(trOp.getInput().getType(), preTransposeAxis);
  SmallVector<Attribute> newPermutation;
  newPermutation.reserve(permutation.size() + 1);
  for (auto [to, from] :
       llvm::enumerate(permutation.getAsRange<IntegerAttr>())) {
    if (to == postTransposeAxis) {
      newPermutation.push_back(from);
      newPermutation.push_back(
          rewriter.getI64IntegerAttr(preTransposeAxis + 1));
    } else if (static_cast<size_t>(from.getInt()) <= preTransposeAxis) {
      newPermutation.push_back(from);
    } else {
      newPermutation.push_back(rewriter.getI64IntegerAttr(from.getInt() + 1));
    }
  }
  Value unpacked = rewriter.create<UnpackOp>(op.getLoc(), preTrCorrectedType,
                                             trOp.getInput(), preTransposeAxis);
  // Not a replaceOpWithNewOp() because we're keeping a different op's location.
  Value transposed = rewriter.create<TransposeOp>(
      trOp.getLoc(), op.getOut().getType(), unpacked,
      rewriter.getArrayAttr(newPermutation));
  rewriter.replaceOp(op, transposed);
  rewriter.eraseOp(trOp);
  return success();
}

LogicalResult ReshapeUnpackInterchange::matchAndRewrite(
    UnpackOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto reshapeOp = adaptor.getIn().getDefiningOp<ReshapeOp>();
  if (!reshapeOp)
    return failure();
  int64_t postReshapeAxis = op.getAxis();
  MIXRShapedType newShapeBytes = op.getIn().getType();
  MIXRShapedType oldShapeBytes = reshapeOp.getInput().getType();

  if (newShapeBytes.getStrides()[postReshapeAxis] != 1)
    return reshapeOp.emitOpError(
               "can't form the int4 tensor type for this value as dimension ")
           << postReshapeAxis
           << " in the new shape should have stride 1 but has "
           << newShapeBytes.getStrides()[postReshapeAxis];
  int64_t lastUnitDim = 0;
  for (auto [idx, stride] : llvm::enumerate(oldShapeBytes.getStrides()))
    if (stride == 1)
      lastUnitDim = idx;
  MIXRShapedType oldShapeInt4 = asInt4Tensor(oldShapeBytes, lastUnitDim);
  MIXRShapedType newShapeInt4 = op.getOut().getType();
  Value unpacked = rewriter.create<UnpackOp>(op.getLoc(), oldShapeInt4,
                                             reshapeOp.getInput());
  Value reshaped = rewriter.create<ReshapeOp>(
      reshapeOp.getLoc(), newShapeInt4, unpacked,
      rewriter.getI64ArrayAttr(newShapeInt4.getShape()));
  rewriter.replaceOp(op, reshaped);
  rewriter.eraseOp(reshapeOp);
  return success();
}

LogicalResult FuncArgUnpackElimination::matchAndRewrite(
    UnpackOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto unpackArg = dyn_cast<BlockArgument>(adaptor.getIn());
  if (!unpackArg)
    return failure();
  auto funcOp =
      dyn_cast<func::FuncOp>(unpackArg.getParentRegion()->getParentOp());
  if (!funcOp)
    return op.emitOpError("A tensor that'll be unpacked is an argument to "
                          "somethng other than a function");
  MIXRShapedType int4Type = op.getResult().getType();
  FunctionType funcType = funcOp.getFunctionType();
  SmallVector<Type> newInTypes(funcType.getInputs());
  newInTypes[unpackArg.getArgNumber()] = int4Type;
  rewriter.modifyOpInPlace(funcOp, [&]() {
    funcOp.setFunctionType(funcType.clone(newInTypes, funcType.getResults()));
    unpackArg.setType(int4Type);
  });
  rewriter.replaceOp(op, unpackArg);
  return success();
}

void MIGraphXRealizeInt4Pass::runOnOperation() {
  func::FuncOp func = getOperation();

  ConversionTarget noPacks(getContext());
  noPacks.addLegalDialect<migraphx::MIGraphXDialect>();
  noPacks.addIllegalOp<migraphx::UnpackOp>();
  noPacks.addLegalOp<func::FuncOp>();

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<TransposeUnpackInterchange, ReshapeUnpackInterchange,
               FuncArgUnpackElimination>(ctx);
  patterns.add<RewriteByteUnpackPattern>(ctx, /*benefit=*/2);

  if (failed(applyPartialConversion(func, noPacks, std::move(patterns))))
    return signalPassFailure();
}
