//===- LowerRockReduce.cpp - The lowering pass of rock.reduce ------------===//
//
// Copyright 2022 Advanced Micro Devices.
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
// This pass converts rock.reduce into the TransformingFor loop
// using global atomics.
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include <memory>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLOWERREDUCEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-lower-rock-reduce"

using namespace mlir;
using namespace mlir::rock;

namespace {
class RockLowerReducePass
    : public rock::impl::RockLowerReducePassBase<RockLowerReducePass> {
  void runOnOperation() override;
};

struct ReduceRewritePattern : public OpConversionPattern<ReduceOp> {
  using OpConversionPattern<ReduceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceOp op, ReduceOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // end namespace

// This function will create a view of the tensor that is being
// reduced from arbitary dimension sizes to bid x tid x iter space.
static ArrayAttr createThreadViewMaps(Value redInput, int64_t blockSize,
                                      int64_t gridSize, Location loc,
                                      PatternRewriter &rewriter) {
  int64_t totalThreads = gridSize * blockSize;
  ShapedType inpShape = redInput.getType().cast<ShapedType>();
  int64_t elementCount = inpShape.getNumElements();
  int64_t dataPerThread = (elementCount + (totalThreads - 1)) / totalThreads;

  SmallVector<int64_t> lowerSizes;
  int64_t lowerSizeProduct = 1;
  SmallVector<unsigned int> lowerDims;
  for (auto dimAndSize : llvm::enumerate(inpShape.getShape())) {
    size_t dim = dimAndSize.index();
    int64_t dimSize = dimAndSize.value();
    lowerSizes.push_back(dimSize);
    lowerSizeProduct *= dimSize;
    lowerDims.push_back(dim);
  }

  BottomUpTMBuilder threadsToInpTensor(rewriter, lowerSizes, loc);
  SmallVector<StringRef, 4> lowerNameRefs;
  threadsToInpTensor.getStartNames(lowerNameRefs);
  threadsToInpTensor.merge("flatDim", 0, lowerNameRefs);
  TransformMapAttr mergeTrMap = threadsToInpTensor.get();

  threadsToInpTensor = BottomUpTMBuilder::above(threadsToInpTensor, mergeTrMap);
  threadsToInpTensor.pad({"flatDim"},
                         {0, totalThreads * dataPerThread - lowerSizeProduct});
  TransformMapAttr padTrMap = threadsToInpTensor.get();

  threadsToInpTensor = BottomUpTMBuilder::above(threadsToInpTensor, padTrMap);
  threadsToInpTensor.unmerge({"bid", "iter", "tid"}, {0, 1, 2}, "flatDim",
                             {gridSize, dataPerThread, blockSize});
  TransformMapAttr unmergeTrMap = threadsToInpTensor.get();

  return rewriter.getArrayAttr({unmergeTrMap, padTrMap, mergeTrMap});
}

static LogicalResult getStoreMethod(ReduceMethod rMethod,
                                    StoreMethod &stMethod) {
  if (rMethod == ReduceMethod::Sum) {
    stMethod = StoreMethod::AtomicAdd;
    return success();
  }
  return failure();
}

LogicalResult ReduceRewritePattern::matchAndRewrite(
    ReduceOp op, ReduceOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  uint64_t redAxis = op.getAxisAttr().getInt();
  uint64_t gridSize = op.getGridSizeAttr().getInt();
  uint64_t blockSize = op.getBlockSizeAttr().getInt();

  ArrayAttr trMaps =
      createThreadViewMaps(op.getIn(), blockSize, gridSize, loc, rewriter);
  ArrayAttr sourceTransformsFromOp;
  Value source;
  std::tie(source, sourceTransformsFromOp) =
      untransform(rewriter, op.getIn(), trMaps);

  ArrayRef<int64_t> threadViewShape =
      trMaps[0].cast<TransformMapAttr>().getUpperBounds();
  int64_t vectorLength =
      getMaxVectorization(sourceTransformsFromOp, /*dim=*/1, threadViewShape[1],
                          source.getType().cast<MemRefType>().getShape());
  SmallVector<int64_t> bounds(threadViewShape.size(), 1LL);
  // Setting iter dimension bounds to threadViewShape size
  bounds[1] = threadViewShape[1];
  SmallVector<int64_t> strides(threadViewShape.size(), 1LL);
  strides[1] = vectorLength;
  Type elementType = source.getType().cast<MemRefType>().getElementType();
  Type vectorType = vectorTypeOrSelf(elementType, vectorLength);

  // Get current workgroup ID.
  WorkgroupIdOp bid =
      rewriter.create<WorkgroupIdOp>(loc, rewriter.getIndexType());
  // Get current workitem ID.
  WorkitemIdOp tid =
      rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());
  Value zeroConstantOp = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value, 3> loadStartCoords = {bid, zeroConstantOp, tid};

  SmallVector<Value> zeroes(threadViewShape.size(), zeroConstantOp);

  TransformingForOp outLoop = rewriter.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{loadStartCoords},
      ArrayRef<Attribute>{sourceTransformsFromOp}, ArrayRef<int64_t>(bounds),
      ArrayRef<int64_t>(strides),
      /*forceUnroll=*/true, /*useIndexDiffs=*/true);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(outLoop.getBody());
    Block::BlockArgListType loadCoords = outLoop.getLowerCoords(/*domain=*/0);
    Value isValid = outLoop.getValidity(/*domain=*/0);
    GlobalLoadOp loadVal =
        rewriter.create<GlobalLoadOp>(loc, vectorType, op.getIn(), isValid, loadCoords);
    Value loadedReg = rewriter.create<GpuAllocOp>(
        loc, MemRefType::get({vectorLength}, elementType, {},
                             gpu::GPUDialect::getPrivateAddressSpace()));
    rewriter.create<InBoundsStoreOp>(loc, loadVal, loadedReg, zeroConstantOp);

    SmallVector<Value, 4> storeCoords;
    for (const auto &idxAndVal : llvm::enumerate(loadCoords)) {
      size_t idx = idxAndVal.index();
      Value val = idxAndVal.value();
      if (idx == redAxis) {
        storeCoords.push_back(zeroConstantOp);
      } else {
        storeCoords.push_back(val);
      }
    }
    StoreMethod stMethod;
    ReduceMethod rMethod = op.getReduceMethod();
    if (getStoreMethod(rMethod, stMethod).failed()) {
      return op.emitError()
             << "The Reduce Method" << getNameForReduceMethod(rMethod)
             << " is not supported.!";
    }
    rewriter.create<GlobalStoreOp>(
        loc, loadedReg, op.getOut(), rewriter.getIndexAttr(vectorLength),
        StoreMethodAttr::get(rewriter.getContext(), stMethod), zeroConstantOp,
        isValid,
        storeCoords);
  }
  rewriter.eraseOp(op);
  return success();
}

void RockLowerReducePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);

  target.addIllegalOp<rock::ReduceOp>();
  target.addLegalDialect<arith::ArithmeticDialect, rock::RockDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<ReduceRewritePattern>(ctx);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
