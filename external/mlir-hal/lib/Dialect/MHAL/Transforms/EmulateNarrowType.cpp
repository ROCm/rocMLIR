//====----- EmulateNarrowType.cpp - Rewrites to handle i4 memory  ----===//
//
// Part of the MHAL Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2024 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/MHAL/Transforms/Passes.h"

#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhal {
#define GEN_PASS_DEF_MHALEMULATENARROWTYPEPASS
#include "mlir/Dialect/MHAL/Transforms/Passes.h.inc"
} // namespace mhal
} // namespace mlir

using namespace mlir;
using namespace mlir::mhal;

namespace {
struct MHalEmulateNarrowTypePass
    : public mhal::impl::MHalEmulateNarrowTypePassBase<
          MHalEmulateNarrowTypePass> {
  void runOnOperation() override;
};
} // end namespace

namespace {
/// When the source memref for the shape expansion is the result of an
/// unrealized_conversion_cast on a block argument (so, in this case, a function
/// argument) and none of the people who extracted the metadata are using the
/// base pointer, just use the information from the pre-conversion type on the
/// theory that it's correct - which, for us, it is, mainly because we've also
/// got a strong presumption of static shapes.
///
/// This is an evil ugly hack of evilness that can go away once we have a
/// defined pointer-based calling convention so that the memrefs are constructed
/// within the function body like they are with IREE
class ExtractStridedMetadataFromOldFuncArgs
    : public OpRewritePattern<memref::ExtractStridedMetadataOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp extractStridedMetadataOp,
                  PatternRewriter &rewriter) const override {
    Location loc = extractStridedMetadataOp.getLoc();
    Value source = extractStridedMetadataOp.getSource();
    MemRefType sourceType = extractStridedMetadataOp.getSource().getType();
    auto castOp = source.getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp)
      return failure();
    if (!extractStridedMetadataOp.getBaseBuffer().use_empty())
      return failure();
    if (!sourceType.hasStaticShape() || !sourceType.getLayout().isIdentity())
      return failure();
    if (!isa<BlockArgument>(castOp.getInputs()[0]))
      return failure();

    unsigned rank = sourceType.getRank();
    ArrayRef<int64_t> shape = sourceType.getShape();
    SmallVector<int64_t> strides = computeStrides(shape);
    SmallVector<Value> results;
    results.reserve(2 * rank + 2);
    results.push_back(nullptr); // base buffer
    auto makeIndexOp = [&](int64_t value) -> Value {
      return rewriter.createOrFold<arith::ConstantIndexOp>(loc, value);
    };
    results.push_back(makeIndexOp(0)); // offset
    llvm::transform(shape, std::back_inserter(results), makeIndexOp);
    llvm::transform(strides, std::back_inserter(results), makeIndexOp);
    rewriter.replaceOp(extractStridedMetadataOp, results);
    return success();
  }
};

struct MHalLaunchOpRewritePattern : public OpConversionPattern<LaunchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LaunchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newReturnTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                newReturnTypes)))
      return rewriter.notifyMatchFailure(
          op, "failed to convert result type of launched function");
    rewriter.replaceOpWithNewOp<LaunchOp>(
        op, newReturnTypes.front(), ArrayRef<Type>(newReturnTypes).drop_front(),
        adaptor.getCallee(), adaptor.getDependencies(),
        adaptor.getLaunchOperands());
    return success();
  }
};
} // end namespace

void mlir::mhal::populateMHalNarrowTypeEmulationConversions(
    arith::NarrowTypeEmulationConverter &typeConverter) {
  // We'll see uses of the old argument in extract_strided_metadata and
  // in ops that haven't been converted yet.
  auto materializer = [](OpBuilder &builder, MemRefType illegalType,
                         ValueRange inputs,
                         Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;
    Value input = inputs.front();
    if (input.getType() == illegalType)
      return input;
    return builder.create<UnrealizedConversionCastOp>(loc, illegalType, input)
        .getResult(0);
  };
  typeConverter.addSourceMaterialization(materializer);
  typeConverter.addTargetMaterialization(materializer);
}

void mlir::mhal::populateMHalNarrowTypeEmulationBoundaryPatterns(
    arith::NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns) {
  patterns.add<MHalLaunchOpRewritePattern>(typeConverter,
                                           patterns.getContext());
}

void mlir::mhal::populateMHalNarrowTypeEmulationPatterns(
    arith::NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns) {
  std::ignore = typeConverter;
  patterns.add<ExtractStridedMetadataFromOldFuncArgs>(patterns.getContext());
}

void MHalEmulateNarrowTypePass::runOnOperation() {
  func::FuncOp op = getOperation();
  MLIRContext *ctx = &getContext();

  // Note that since this is meant to be run on test code, it doesn't handle
  // `vector`.

  arith::NarrowTypeEmulationConverter typeConverter(/*targetBitwidth=*/8);
  memref::populateMemRefNarrowTypeEmulationConversions(typeConverter);
  mhal::populateMHalNarrowTypeEmulationConversions(typeConverter);

  auto opLegalCallback = [&typeConverter](Operation *op) {
    return typeConverter.isLegal(op);
  };
  ConversionTarget boundaryTarget(*ctx);
  boundaryTarget.addDynamicallyLegalOp<func::FuncOp>(
      [&typeConverter](func::FuncOp op) {
        return typeConverter.isLegal(op.getFunctionType());
      });
  boundaryTarget.addDynamicallyLegalOp<mhal::LaunchOp>(
      [&typeConverter](mhal::LaunchOp op) {
        return typeConverter.isLegal(op.getCallResultTypes()) &&
               typeConverter.isLegal(op.getOperandTypes());
      });
  boundaryTarget.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
      opLegalCallback);

  ConversionTarget target(*ctx);
  target.addDynamicallyLegalDialect<memref::MemRefDialect,
                                    affine::AffineDialect, arith::ArithDialect>(
      opLegalCallback);

  // First, we do the conversions on function signatures so that we can get some
  // unrealized_conversion_cast ops that'll cancel out later.
  RewritePatternSet boundaryPatterns(ctx);
  arith::populateArithNarrowTypeEmulationPatterns(typeConverter,
                                                  boundaryPatterns);
  mhal::populateMHalNarrowTypeEmulationBoundaryPatterns(typeConverter,
                                                        boundaryPatterns);
  if (failed(applyPartialConversion(op, boundaryTarget,
                                    std::move(boundaryPatterns))))
    return signalPassFailure();

  RewritePatternSet patterns(ctx);
  memref::populateMemRefNarrowTypeEmulationPatterns(typeConverter, patterns);
  mhal::populateMHalNarrowTypeEmulationPatterns(typeConverter, patterns);
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return signalPassFailure();
}
