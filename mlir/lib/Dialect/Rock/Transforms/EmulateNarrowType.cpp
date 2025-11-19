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

#include "mlir/IR/Matchers.h"
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
      return rewriter.notifyMatchFailure(op,
                                         "couldn't convert allocation type");
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
      return arith::ConstantIndexOp::create(rewriter, loc, value);
    };
    Value baseBuffer = nullptr;
    if (!op.getBaseBuffer().use_empty()) {
      baseBuffer = memref::ReinterpretCastOp::create(
          rewriter, loc, cast<MemRefType>(op.getBaseBuffer().getType()),
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

struct ExtractStridedMetadataOpMemRefViewFolder
    : public OpRewritePattern<memref::ExtractStridedMetadataOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExtractStridedMetadataOp op,
                                PatternRewriter &rewriter) const override {
    auto viewOp = op.getSource().getDefiningOp<memref::ViewOp>();
    if (!viewOp)
      return failure();

    Location loc = op.getLoc();
    auto makeConst = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantIndexOp>(loc, value);
    };

    // For memref.view, the base buffer is the source memref
    Value baseBuffer = viewOp.getSource();

    // The offset is the byte shift from the view operation
    Value offset = viewOp.getByteShift();

    // For a 1D view (which Rock uses), size is the number of elements
    MemRefType viewType = viewOp.getType();
    if (viewType.getRank() != 1)
      return failure(); // Only handle 1D views for now

    Value size = makeConst(viewType.getNumElements());
    Value stride = makeConst(1); // Contiguous layout

    // If the base buffer is needed, reinterpret cast it to the expected type
    if (!op.getBaseBuffer().use_empty()) {
      baseBuffer = rewriter.create<memref::ReinterpretCastOp>(
          loc, cast<MemRefType>(op.getBaseBuffer().getType()), baseBuffer, 0,
          ArrayRef<int64_t>(), ArrayRef<int64_t>());
    }

    SmallVector<Value, 4> results = {baseBuffer, offset, size, stride};
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct FatRawBufferCastRewritePattern
    : public OpConversionPattern<amdgpu::FatRawBufferCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(amdgpu::FatRawBufferCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType oldType = op.getResult().getType();
    Type newType = getTypeConverter()->convertType(oldType);
    if (!newType)
      return rewriter.notifyMatchFailure(op, "failed to convert type");
    if (oldType == newType)
      return rewriter.notifyMatchFailure(
          op, "type unchanged, no conversion needed");

    rewriter.replaceOpWithNewOp<amdgpu::FatRawBufferCastOp>(
        op, newType, adaptor.getSource(), adaptor.getValidBytes(),
        adaptor.getCacheSwizzleStride(), adaptor.getBoundsCheckAttr(),
        adaptor.getResetOffsetAttr());
    return success();
  }
};

struct MemRefViewRewritePattern : public OpConversionPattern<memref::ViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType oldType = op.getResult().getType();

    // Convert the type through the type converter
    Type newType = getTypeConverter()->convertType(oldType);
    if (!newType)
      return rewriter.notifyMatchFailure(op,
                                         Twine("couldn't convert view type"));
    if (oldType == newType)
      return failure();

    // Create a new view with the converted type
    // The source buffer should already be converted by other patterns
    rewriter.replaceOpWithNewOp<memref::ViewOp>(
        op, newType, adaptor.getSource(), adaptor.getByteShift(),
        adaptor.getSizes());
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
    Value nibbleIndex = adaptor.getIndices()[0];
    unsigned indexWidth = cast<IntegerType>(nibbleIndex.getType()).getWidth();
    Value scaleConst =
        rewriter.createOrFold<arith::ConstantIntOp>(loc, scale, indexWidth);
    Value newIndex =
        arith::DivUIOp::create(rewriter, loc, nibbleIndex, scaleConst);

    auto convertedMemrefType =
        dyn_cast<MemRefType>(adaptor.getMemref().getType());
    auto origMemrefType = dyn_cast<MemRefType>(op.getMemref().getType());
    int64_t numBytes = convertedMemrefType.getNumElements();
    // if the original out of bound index was odd then need to clamp to the next
    // byte to make sure it stays out of bounds
    int64_t numNibbles = origMemrefType.getNumElements();
    Value numBytesConst =
        rewriter.createOrFold<arith::ConstantIntOp>(loc, numBytes, indexWidth);
    Value nibbleBoundConst = rewriter.createOrFold<arith::ConstantIntOp>(
        loc, numNibbles, indexWidth);

    Value nibbleIsOob = rewriter.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, nibbleIndex, nibbleBoundConst);
    newIndex = rewriter.createOrFold<arith::SelectOp>(loc, nibbleIsOob,
                                                      numBytesConst, newIndex);
    // Note: if you're using sgpr offset for some reason, this won't work.
    Value newLoad = amdgpu::RawBufferLoadOp::create(
        rewriter, loc, newType, adaptor.getMemref(), newIndex,
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
    Value newIndex = arith::DivUIOp::create(
        rewriter, loc, adaptor.getIndices()[0], scaleConst);
    // Note: if you're using sgpr offset for some reason, this won't work.
    Value newValue =
        vector::BitCastOp::create(rewriter, loc, newType, adaptor.getValue());
    rewriter.replaceOpWithNewOp<amdgpu::RawBufferStoreOp>(
        op, newValue, adaptor.getMemref(), newIndex,
        adaptor.getBoundsCheckAttr(), nullptr, nullptr);
    return success();
  }
};

struct GatherToLDSRewritePattern
    : public OpConversionPattern<amdgpu::GatherToLDSOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(amdgpu::GatherToLDSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *converter = getTypeConverter<arith::NarrowTypeEmulationConverter>();

    auto origSrcMemrefType = dyn_cast<MemRefType>(op.getSrc().getType());
    auto origDstMemrefType = dyn_cast<MemRefType>(op.getDst().getType());

    if (!origSrcMemrefType || !origDstMemrefType)
      return rewriter.notifyMatchFailure(op, "expected memref types");

    // Check if we're dealing with narrow types
    unsigned oldWidthSrc = origSrcMemrefType.getElementTypeBitWidth();
    unsigned oldWidthDst = origDstMemrefType.getElementTypeBitWidth();
    if (oldWidthSrc != 4 || oldWidthDst != 4)
      return rewriter.notifyMatchFailure(
          op, "expected 4-bit element types for source and destination for "
              "GatherToLDS emulation");

    unsigned newWidth = converter->getLoadStoreBitwidth();
    unsigned scale = newWidth / oldWidthSrc;

    auto convertedSrcMemrefType =
        dyn_cast<MemRefType>(adaptor.getSrc().getType());
    auto convertedDstMemrefType =
        dyn_cast<MemRefType>(adaptor.getDst().getType());
    assert(newWidth == 8 &&
           "expected 8-bit element types for narrow type emulation");
    int64_t numSrcBytes = convertedSrcMemrefType.getNumElements();
    int64_t numDstBytes = convertedDstMemrefType.getNumElements();
    int64_t numSrcNibbles = origSrcMemrefType.getNumElements();
    int64_t numDstNibbles = origDstMemrefType.getNumElements();
    // Check that transferType is f128 or f32 as currently only 128bit or 32 bit
    // DirectToLDS is supported.
    Type transferType = op.getTransferType();
    if (!transferType.isF128() && !transferType.isF32()) {
      return op->emitError("Transfer type must be f128 or f32 for GatherToLDS");
    }
    // Helper lambda to scale indices for 4-bit type emulation
    auto scaleIndexForNarrowType = [&](Value lastIndex, int64_t numBytes,
                                       int64_t numNibbles,
                                       StringRef errorMsg) -> FailureOr<Value> {
      // Check if the last index is even (divisible by 2)
      // It is possible that this is not a compile time constant and it cannot
      // be matched. Doing runtime checks is very expensive. Given directToLDS
      // transfers 128 bit or 32 bits, it should be safe to assume that last
      // co-ord will be even at runtime.
      APInt indexConst(64, 0);
      if (matchPattern(lastIndex, m_ConstantInt(&indexConst))) {
        if (indexConst.urem(2) != 0) {
          return rewriter.notifyMatchFailure(op, errorMsg);
        }
      }

      Type indexType = lastIndex.getType();
      bool isIndexType = isa<IndexType>(indexType);
      Value scaleConst;
      Value numBytesConst;
      Value nibbleBoundConst;

      if (isIndexType) {
        scaleConst = rewriter.create<arith::ConstantIndexOp>(loc, scale);
        numBytesConst = rewriter.create<arith::ConstantIndexOp>(loc, numBytes);
        nibbleBoundConst =
            rewriter.create<arith::ConstantIndexOp>(loc, numNibbles);
      } else {
        unsigned indexWidth = cast<IntegerType>(indexType).getWidth();
        scaleConst =
            rewriter.createOrFold<arith::ConstantIntOp>(loc, scale, indexWidth);
        numBytesConst = rewriter.createOrFold<arith::ConstantIntOp>(
            loc, numBytes, indexWidth);
        nibbleBoundConst = rewriter.createOrFold<arith::ConstantIntOp>(
            loc, numNibbles, indexWidth);
      }
      Value scaledIndex =
          arith::DivUIOp::create(rewriter, loc, lastIndex, scaleConst);

      // Handle OOB: if the original nibble index was odd and OOB, clamp to next
      // byte to ensure it stays out of bounds
      Value nibbleIsOob = rewriter.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::uge, lastIndex, nibbleBoundConst);
      scaledIndex = rewriter.createOrFold<arith::SelectOp>(
          loc, nibbleIsOob, numBytesConst, scaledIndex);

      return scaledIndex;
    };

    // Scale source indices (last index)
    SmallVector<Value> newSrcIndices(adaptor.getSrcIndices());
    if (!newSrcIndices.empty()) {
      auto scaledIndexOrFailure = scaleIndexForNarrowType(
          newSrcIndices.back(), numSrcBytes, numSrcNibbles,
          "last source index must be even for 4-bit type emulation");
      if (failed(scaledIndexOrFailure))
        return failure();
      newSrcIndices.back() = scaledIndexOrFailure.value();
    }

    // Scale destination indices (last index)
    SmallVector<Value> newDstIndices(adaptor.getDstIndices());
    if (!newDstIndices.empty()) {
      auto scaledIndexOrFailure = scaleIndexForNarrowType(
          newDstIndices.back(), numDstBytes, numDstNibbles,
          "last destination index must be even for 4-bit type emulation");
      if (failed(scaledIndexOrFailure))
        return failure();
      newDstIndices.back() = scaledIndexOrFailure.value();
    }

    // Create new GatherToLDSOp with scaled indices
    rewriter.replaceOpWithNewOp<amdgpu::GatherToLDSOp>(
        op, adaptor.getSrc(), newSrcIndices, adaptor.getDst(), newDstIndices,
        op.getTransferTypeAttr());

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
  if (failed(applyPatternsGreedily(op, std::move(prePatterns))))
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
               BufferStoreRewritePatttern, GatherToLDSRewritePattern,
               FatRawBufferCastRewritePattern, MemRefViewRewritePattern>(
      typeConverter, ctx);
  patterns.add<ExtractStridedMetadataOpRockGpuAllocFolder,
               ExtractStridedMetadataOpMemRefViewFolder>(ctx);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return signalPassFailure();

  RewritePatternSet postPatterns(ctx);
  vector::populateVectorNarrowTypeRewritePatterns(postPatterns);
  if (failed(applyPatternsGreedily(op, std::move(postPatterns))))
    return signalPassFailure();
}
