//===- EmulateNarrowType.cpp - Rewrite 4-bit memrefs -- ------------===//
//
// Copyright 2024 Advanced Micro Devices.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================
//
// This pass coordinates the upstream rewrite rules for 4-bit types, ensuring we
// apply just those we want.
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/MHAL/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"

#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKEMULATENARROWTYPEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-emulate-narrow-type"

using namespace mlir;

namespace {
class RockEmulateNarrowTypePass
    : public rock::impl::RockEmulateNarrowTypePassBase<
          RockEmulateNarrowTypePass> {
  void runOnOperation() override;
};
} // end namespace

namespace {
struct RockGpuAllocRewritePattern
    : public OpConversionPattern<rock::GpuAllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(rock::GpuAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType oldType = op.getResult().getType();
    Type newType = getTypeConverter()->convertType(oldType);
    if (!newType)
      return rewriter.notifyMatchFailure(
          op, Twine("couldn't convert allocation type"));
    if (oldType.getRank() != 1)
      return rewriter.notifyMatchFailure(
          op, "expected 1-D int4 tiles in our internals");
    if (oldType.getNumElements() % 2 != 0)
      return rewriter.notifyMatchFailure(
          op, "expected internal buffers to use an even number of bytes");
    rewriter.replaceOpWithNewOp<rock::GpuAllocOp>(op, newType);
    return success();
  }
};

// Impmentent extract_strided_metadata for gpu.alloc ops as
// baseBuffer = [the base buffer]
// offset = 0
// sizes = [the allocation size]
// strides = {1}
struct ExtractStridedMetadataOpRockGpuAllocFolder
    : public OpRewritePattern<memref::ExtractStridedMetadataOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExtractStridedMetadataOp op,
                                PatternRewriter &rewriter) const override {
    auto gpuAlloc = op.getSource().getDefiningOp<rock::GpuAllocOp>();
    if (!gpuAlloc)
      return failure();
    Location loc = op.getLoc();
    auto makeConst = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantIndexOp>(loc, value);
    };
    Value baseBuffer = nullptr;
    if (!op.getBaseBuffer().use_empty()) {
      baseBuffer = rewriter.create<memref::ReinterpretCastOp>(
          loc, cast<MemRefType>(op.getBaseBuffer().getType()),
          gpuAlloc.getResult(), 0, ArrayRef<int64_t>(), ArrayRef<int64_t>());
    }
    SmallVector<Value, 4> results = {
        gpuAlloc.getResult(), makeConst(0),
        makeConst(gpuAlloc.getResult().getType().getNumElements()),
        makeConst(1)};
    rewriter.replaceOp(op, results);
    return success();
  }
};
} // namespace

namespace {
/// Temporary workarounds for not using addrspace(7)
struct BufferLoadRewritePatttern
    : public OpConversionPattern<amdgpu::RawBufferLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(amdgpu::RawBufferLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto oldType = dyn_cast<VectorType>(op.getResult().getType());
    auto *converter = getTypeConverter<arith::NarrowTypeEmulationConverter>();
    if (!oldType)
      return rewriter.notifyMatchFailure(
          op, "expected vector i4 loads in temp code");
    if (op.getMemref().getType().getRank() != 1)
      return rewriter.notifyMatchFailure(
          op, "expect linearized indices because it's rock");
    unsigned oldWidth = oldType.getElementTypeBitWidth();
    unsigned newWidth = converter->getLoadStoreBitwidth();
    unsigned scale = newWidth / oldWidth;
    if (oldType.getNumElements() % scale != 0)
      return rewriter.notifyMatchFailure(
          op, "expected even buffer load in temp code");
    auto newType = VectorType::get(oldType.getNumElements() / scale,
                                   rewriter.getIntegerType(newWidth));
    Value scaleConst =
        rewriter.createOrFold<arith::ConstantIntOp>(loc, scale, /*width=*/32);
    Value newIndex = rewriter.create<arith::DivUIOp>(
        loc, adaptor.getIndices()[0], scaleConst);
    // Note: if you're using sgpr offset for some reason, this won't work.
    Value newLoad = rewriter.create<amdgpu::RawBufferLoadOp>(
        loc, newType, adaptor.getMemref(), newIndex,
        adaptor.getBoundsCheckAttr(), nullptr, nullptr);
    rewriter.replaceOpWithNewOp<vector::BitCastOp>(op, oldType, newLoad);
    return success();
  }
};

struct BufferStoreRewritePatttern
    : public OpConversionPattern<amdgpu::RawBufferStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(amdgpu::RawBufferStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto oldType = dyn_cast<VectorType>(op.getValue().getType());
    auto *converter = getTypeConverter<arith::NarrowTypeEmulationConverter>();
    if (!oldType)
      return rewriter.notifyMatchFailure(
          op, "expected vector i4 loads in temp code");
    if (op.getMemref().getType().getRank() != 1)
      return rewriter.notifyMatchFailure(
          op, "expect linearized indices because it's rock");
    unsigned oldWidth = oldType.getElementTypeBitWidth();
    unsigned newWidth = converter->getLoadStoreBitwidth();
    unsigned scale = newWidth / oldWidth;
    if (oldType.getNumElements() % scale != 0)
      return rewriter.notifyMatchFailure(
          op, "expected even buffer load in temp code");
    auto newType = VectorType::get(oldType.getNumElements() / scale,
                                   rewriter.getIntegerType(newWidth));
    Value scaleConst =
        rewriter.createOrFold<arith::ConstantIntOp>(loc, scale, /*width=*/32);
    Value newIndex = rewriter.create<arith::DivUIOp>(
        loc, adaptor.getIndices()[0], scaleConst);
    // Note: if you're using sgpr offset for some reason, this won't work.
    Value newValue =
        rewriter.create<vector::BitCastOp>(loc, newType, adaptor.getValue());
    rewriter.replaceOpWithNewOp<amdgpu::RawBufferStoreOp>(
        op, newValue, adaptor.getMemref(), newIndex,
        adaptor.getBoundsCheckAttr(), nullptr, nullptr);
    return success();
  }
};
} // end namespace

void RockEmulateNarrowTypePass::runOnOperation() {
  func::FuncOp op = getOperation();
  MLIRContext *ctx = &getContext();

  RewritePatternSet prePatterns(ctx);
  // (Needed because the lowering on rank-0 tensors creates vector.broadcasts,
  // which LLVM conversion doesn't understand).
  vector::populateVectorBroadcastLoweringPatterns(prePatterns);
  // Lower vector.transfer_{read,write} to vector.{load,store} here to let the
  // vector patterns go through.
  vector::populateVectorTransferLoweringPatterns(prePatterns,
                                                 /*maxTransferRank=*/1);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(prePatterns))))
    return signalPassFailure();

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
  boundaryTarget.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
      opLegalCallback);

  ConversionTarget target(*ctx);
  target.addDynamicallyLegalDialect<
      vector::VectorDialect, memref::MemRefDialect, amdgpu::AMDGPUDialect,
      affine::AffineDialect, arith::ArithDialect>(opLegalCallback);
  target.addDynamicallyLegalOp<rock::GpuAllocOp>(opLegalCallback);

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
  vector::populateVectorNarrowTypeEmulationPatterns(typeConverter, patterns);
  mhal::populateMHalNarrowTypeEmulationPatterns(typeConverter, patterns);
  patterns.add<RockGpuAllocRewritePattern, BufferLoadRewritePatttern,
               BufferStoreRewritePatttern>(typeConverter, ctx);
  patterns.add<ExtractStridedMetadataOpRockGpuAllocFolder>(ctx);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return signalPassFailure();

  RewritePatternSet postPatterns(ctx);
  vector::populateVectorNarrowTypeRewritePatterns(postPatterns);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(postPatterns))))
    return signalPassFailure();
}
