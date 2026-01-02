//===- LowerScatterGather.cpp - Lower rock.scatter and rock.gather -------===//
//
// Copyright 2025 Advanced Micro Devices.
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
// This pass lowers rock.scatter and rock.gather operations to Rock threadwise
// ops that integrate with the GPU threading model.
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLOWERSCATTERGATHERPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-lower-scatter-gather"

using namespace mlir;
using namespace mlir::rock;

namespace {

class RockLowerScatterGatherPass
    : public rock::impl::RockLowerScatterGatherPassBase<
          RockLowerScatterGatherPass> {
  void runOnOperation() override;
};

/// Get block_size from function attributes
static int64_t getBlockSize(func::FuncOp func) {
  if (auto attr = func->getAttrOfType<IntegerAttr>("block_size"))
    return attr.getInt();
  // Default block size if not specified
  return 256;
}

//===----------------------------------------------------------------------===//
// ScatterOp lowering
//===----------------------------------------------------------------------===//

struct ScatterRewritePattern : public OpConversionPattern<ScatterOp> {
  using OpConversionPattern<ScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ScatterOp op, ScatterOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value cache = adaptor.getCache();
    Value indices = adaptor.getIndices();
    Value updates = adaptor.getUpdates();

    auto indicesType = cast<MemRefType>(indices.getType());
    auto updatesType = cast<MemRefType>(updates.getType());

    // Shape: cache [batch, seqLen, hidden], indices [batch, numUpdates],
    //        updates [batch, numUpdates, hidden]
    int64_t batchSize = indicesType.getShape()[0];
    int64_t numUpdates = indicesType.getShape()[1];
    int64_t hiddenSize = updatesType.getShape()[2];

    // Get block size from parent function
    auto func = op->getParentOfType<func::FuncOp>();
    int64_t blockSize = getBlockSize(func);

    // Total work items = batch * numUpdates * hidden
    int64_t totalWork = batchSize * numUpdates * hiddenSize;
    // Work per thread (ceiling division)
    int64_t workPerThread = llvm::divideCeil(totalWork, blockSize);

    // Get thread ID using Rock's workitem_id
    Value tid = rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());

    // Create constants
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value blockSizeVal =
        rewriter.create<arith::ConstantIndexOp>(loc, blockSize);
    Value totalWorkVal =
        rewriter.create<arith::ConstantIndexOp>(loc, totalWork);
    Value hiddenSizeVal =
        rewriter.create<arith::ConstantIndexOp>(loc, hiddenSize);
    Value numUpdatesTimesHidden =
        rewriter.create<arith::ConstantIndexOp>(loc, numUpdates * hiddenSize);

    // Each thread processes elements: tid, tid + blockSize, tid + 2*blockSize,
    // ...
    rewriter.create<scf::ForOp>(
        loc, tid, totalWorkVal, blockSizeVal, ValueRange{},
        [&](OpBuilder &b, Location forLoc, Value linearIdx, ValueRange) {
          // Decompose linear index into (batch, update, hidden)
          // linearIdx = batch * (numUpdates * hidden) + update * hidden +
          // hidden_idx
          Value batchIdx =
              b.create<arith::DivUIOp>(forLoc, linearIdx, numUpdatesTimesHidden);
          Value remainder =
              b.create<arith::RemUIOp>(forLoc, linearIdx, numUpdatesTimesHidden);
          Value updateIdx =
              b.create<arith::DivUIOp>(forLoc, remainder, hiddenSizeVal);
          Value hiddenIdx =
              b.create<arith::RemUIOp>(forLoc, remainder, hiddenSizeVal);

          // Load the cache index: cachePos = indices[batch, updateIdx]
          Value cachePos = b.create<memref::LoadOp>(
              forLoc, indices, ValueRange{batchIdx, updateIdx});

          // Convert index from i32 to index type
          Value cachePosIndex = b.create<arith::IndexCastOp>(
              forLoc, b.getIndexType(), cachePos);

          // Load value from updates: val = updates[batch, updateIdx, hidden]
          Value val = b.create<memref::LoadOp>(
              forLoc, updates, ValueRange{batchIdx, updateIdx, hiddenIdx});

          // Store to cache: cache[batch, cachePos, hidden] = val
          b.create<memref::StoreOp>(
              forLoc, val, cache,
              ValueRange{batchIdx, cachePosIndex, hiddenIdx});

          b.create<scf::YieldOp>(forLoc);
        });

    // Add LDS barrier after scatter to synchronize all threads
    rewriter.create<LDSBarrierOp>(loc);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GatherOp lowering
//===----------------------------------------------------------------------===//

struct GatherRewritePattern : public OpConversionPattern<GatherOp> {
  using OpConversionPattern<GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GatherOp op, GatherOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value cache = adaptor.getCache();
    Value indices = adaptor.getIndices();
    Value out = adaptor.getOut();

    auto indicesType = cast<MemRefType>(indices.getType());
    auto outType = cast<MemRefType>(out.getType());

    // Shape: cache [batch, seqLen, hidden], indices [batch, numGathers],
    //        out [batch, numGathers, hidden]
    int64_t batchSize = indicesType.getShape()[0];
    int64_t numGathers = indicesType.getShape()[1];
    int64_t hiddenSize = outType.getShape()[2];

    // Get block size from parent function
    auto func = op->getParentOfType<func::FuncOp>();
    int64_t blockSize = getBlockSize(func);

    // Total work items = batch * numGathers * hidden
    int64_t totalWork = batchSize * numGathers * hiddenSize;

    // Get thread ID using Rock's workitem_id
    Value tid = rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());

    // Create constants
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value blockSizeVal =
        rewriter.create<arith::ConstantIndexOp>(loc, blockSize);
    Value totalWorkVal =
        rewriter.create<arith::ConstantIndexOp>(loc, totalWork);
    Value hiddenSizeVal =
        rewriter.create<arith::ConstantIndexOp>(loc, hiddenSize);
    Value numGathersTimesHidden =
        rewriter.create<arith::ConstantIndexOp>(loc, numGathers * hiddenSize);

    // Each thread processes elements: tid, tid + blockSize, tid + 2*blockSize,
    // ...
    rewriter.create<scf::ForOp>(
        loc, tid, totalWorkVal, blockSizeVal, ValueRange{},
        [&](OpBuilder &b, Location forLoc, Value linearIdx, ValueRange) {
          // Decompose linear index into (batch, gather, hidden)
          Value batchIdx =
              b.create<arith::DivUIOp>(forLoc, linearIdx, numGathersTimesHidden);
          Value remainder =
              b.create<arith::RemUIOp>(forLoc, linearIdx, numGathersTimesHidden);
          Value gatherIdx =
              b.create<arith::DivUIOp>(forLoc, remainder, hiddenSizeVal);
          Value hiddenIdx =
              b.create<arith::RemUIOp>(forLoc, remainder, hiddenSizeVal);

          // Load the cache index: cachePos = indices[batch, gatherIdx]
          Value cachePos = b.create<memref::LoadOp>(
              forLoc, indices, ValueRange{batchIdx, gatherIdx});

          // Convert index from i32 to index type
          Value cachePosIndex = b.create<arith::IndexCastOp>(
              forLoc, b.getIndexType(), cachePos);

          // Load from cache: val = cache[batch, cachePos, hidden]
          Value val = b.create<memref::LoadOp>(
              forLoc, cache, ValueRange{batchIdx, cachePosIndex, hiddenIdx});

          // Store to output: out[batch, gatherIdx, hidden] = val
          b.create<memref::StoreOp>(forLoc, val, out,
                                    ValueRange{batchIdx, gatherIdx, hiddenIdx});

          b.create<scf::YieldOp>(forLoc);
        });

    // Add LDS barrier after gather to synchronize all threads
    rewriter.create<LDSBarrierOp>(loc);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

void RockLowerScatterGatherPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  func::FuncOp func = getOperation();

  ConversionTarget target(*ctx);
  target.addIllegalOp<ScatterOp, GatherOp>();
  target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                         memref::MemRefDialect, rock::RockDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<ScatterRewritePattern, GatherRewritePattern>(ctx);

  if (failed(applyPartialConversion(func, target, std::move(patterns))))
    signalPassFailure();
}

} // end anonymous namespace

