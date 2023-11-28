//===----------- MultiBuffering.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements multi buffering transformation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Transforms/RockMultibuffer.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <deque>
#include <set>

using namespace mlir;
using namespace mlir::rock;

#define DEBUG_TYPE "rock-multibuffer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace {
// Set of utility functions used during the transformation

/// Find all the final users of a rockAllocOp. By final users we mean
/// operations that write/load data from the alloc
void findAllocUsers(rock::GpuAllocOp allocOp,
                    SmallPtrSet<Operation *, 16> &users) {
  std::vector<Operation *> worklist{allocOp.getOperation()};

  while (!worklist.empty()) {
    Operation *curOp = worklist.back();
    worklist.pop_back();
    for (Operation *user : curOp->getUsers()) {
      if (dyn_cast<ViewLikeOpInterface>(user)) {
        worklist.push_back(user);
      } else if (!users.contains(user)) {
        users.insert(user);
      }
    }
  }
}

/// Given an original type `<dim x type>`, return its multibuffer version.
/// - If isFlat is true, this will be `<multiBufferingFactor*dim x type>`
/// - If isFlat is false, this will be `<multiBufferingFactor x dim x type>`
MemRefType getFlatMultiBufferType(MemRefType originalType,
                                  int64_t multiBufferingFactor) {
  ArrayRef<int64_t> originalShape = originalType.getShape();
  SmallVector<int64_t, 4> multiBufferedShape;
  assert(originalShape.size() == 1 && "Original buffers need to be flat");
  multiBufferedShape.push_back(multiBufferingFactor * originalShape.back());
  MemRefType mbMemRefType = MemRefType::Builder(originalType)
                                .setShape(multiBufferedShape)
                                .setLayout(MemRefLayoutAttrInterface());
  return mbMemRefType;
}

/// This function goes through the stack of tansforms to add
/// and additional PassThrough transform to allow the multibuffer
/// index to be routed through the stack
TransformMapAttr getMultiBufferTransform(MLIRContext *ctx,
                                         TransformMapAttr transform,
                                         int64_t loopLength) {
  auto ops = llvm::to_vector(transform.getOps());
  SmallVector<TransformAttr> newOps;
  // Since we are adding a PassThrough as first transformation
  // we need to shift all the dimensions of the other transformations
  // by one
  for (auto t : ops) {
    auto lowerDims = llvm::to_vector(t.getLowerDims());
    auto upperDims = llvm::to_vector(t.getUpperDims());
    for (auto &d : lowerDims) {
      d++;
    }
    for (auto &d : upperDims) {
      d++;
    }
    newOps.push_back(TransformAttr::get(ctx, t.getType(), t.getParams(),
                                        t.getUpperNames(), upperDims,
                                        t.getLowerNames(), lowerDims));
  }

  // Create and add the passthrough transform. Notice that logically
  // every loop will have its own multibuffer. We already added
  // a broadcast transformation on top of the transform stack
  // to route the indices back to the real (smaller) multibuffer
  SmallVector<int64_t, 4> multiBufferedUpperBounds{loopLength};
  SmallVector<int64_t, 4> multiBufferedLowerBounds{loopLength};
  ArrayRef<int64_t> originalUpperBounds = transform.getUpperBounds();
  ArrayRef<int64_t> originalLowerBounds = transform.getLowerBounds();
  llvm::append_range(multiBufferedUpperBounds, originalUpperBounds);
  llvm::append_range(multiBufferedLowerBounds, originalLowerBounds);
  auto passThrough =
      TransformAttr::get(ctx, TransformType::PassThrough, {},
                         {"multi_buffer_idx"}, {0}, {"multi_buffer_idx"}, {0});
  newOps.push_back(passThrough);
  TransformMapAttr newMap = TransformMapAttr::get(
      newOps, multiBufferedUpperBounds, multiBufferedLowerBounds);
  return newMap;
}

/// Helper function to create a memref.subview to extract the multibuffer
/// slice.
Value getSubviewIntoMultibuffer(RewriterBase &rewriter, Location loc, Value val,
                                Value index, ArrayRef<int64_t> originalShape,
                                MemRefType mbType) {
  int64_t mbMemRefTypeRank = mbType.getRank();
  IntegerAttr zero = rewriter.getIndexAttr(0);
  IntegerAttr one = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> offsets(mbMemRefTypeRank, zero);
  SmallVector<OpFoldResult> sizes(mbMemRefTypeRank, one);
  SmallVector<OpFoldResult> strides(mbMemRefTypeRank, one);
  // Offset is [bufferIndex, 0 ... 0 ].
  offsets.front() = index;
  // Sizes is [1, original_size_0 ... original_size_n ].
  for (int64_t i = 0, e = originalShape.size(); i != e; ++i)
    sizes[1 + i] = rewriter.getIndexAttr(originalShape[i]);
  // Strides is [1, 1 ... 1 ].
  auto dstMemref =
      cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          originalShape, mbType, offsets, sizes, strides));
  Value subview = rewriter.create<memref::SubViewOp>(loc, dstMemref, val,
                                                     offsets, sizes, strides);
  return subview;
}

/// This is replacing a generic operation op(buffer) with a sequence of
/// - multibuffer_view = memref.subview(buffer)
/// - op(multibuffer_view)
Operation *subviewAndClone(RewriterBase &rewriter, Location loc,
                           OpOperand &operand, Value val, Value loopIndex,
                           int64_t multiBufferingFactor) {
  int64_t operandNumber = operand.getOperandNumber();
  Operation *op = operand.getOwner();
  auto operands = op->getOperands();
  Value buffer = operands[operand.getOperandNumber()];
  auto originalShape = buffer.getType().cast<ShapedType>().getShape();
  MemRefType mbMemRefType = val.getType().cast<MemRefType>();
  auto subview = getSubviewIntoMultibuffer(rewriter, loc, val, loopIndex,
                                           originalShape, mbMemRefType);
  operands[operandNumber] = subview;
  Operation *newOp = rewriter.clone(*op);
  newOp->setOperand(operandNumber, subview);
  return newOp;
}

/// Return if the operation accepts views for operand `operandNumber`
bool acceptsViewAt(Operation *op, OpOperand &operand) {
  auto viewAcceptingOp = dyn_cast<RockAcceptingViewOpInterface>(op);
  if (!viewAcceptingOp)
    return false;
  auto acceptingViewOperands = viewAcceptingOp.getAcceptingViewOperands();
  return acceptingViewOperands.contains(&operand);
}

/// Return true if the `transformInput` is the first of the transform stack
bool isBottomOfTransformStack(Value transformInput) {
  if (dyn_cast<TransformOp>(transformInput.getDefiningOp())) {
    return false;
  }
  return true;
}

/// Get the multibuffer index %mb_i from the target loop
/// mb_i = floor((%i - lowerBound)/step) % multiBufferingFactor
/// This index is needed if we want to memref.subview into the multi-buffer
static Value getMultiBufferIndex(RewriterBase &rewriter, Location loc,
                                 LoopLikeOpInterface loop,
                                 int64_t multiBufferingFactor) {
  std::optional<Value> inductionVar = loop.getSingleInductionVar();
  std::optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
  std::optional<OpFoldResult> singleStep = loop.getSingleStep();

  if (!inductionVar || !lowerBound || !singleStep) {
    llvm_unreachable("Loop variables should already have been tested.");
  }
  Value lbVal = getValueOrCreateConstantIndexOp(rewriter, loc, *lowerBound);
  Value stepVal = getValueOrCreateConstantIndexOp(rewriter, loc, *singleStep);
  Value ivVal = *inductionVar;
  AffineExpr iv, lb, step;
  bindDims(rewriter.getContext(), iv, lb, step);
  Value bufferIndex = affine::makeComposedAffineApply(
      rewriter, loc, ((iv - lb).floorDiv(step)) % multiBufferingFactor,
      {ivVal, lbVal, stepVal});
  LLVM_DEBUG(DBGS() << "--multi-buffered indexing: " << bufferIndex << "\n");
  return bufferIndex;
}

/// Return true if the op is a writer that fully overwrites
/// the given raw `buffer` allocated
bool overrideBuffer(Operation *op, Value buffer) {

  Value dest;
  for (Value operand : op->getOperands()) {
    if (hasEffect<MemoryEffects::Write>(op, operand)) {
      dest = operand;
    }
  }
  if (!dest)
    return false;
  auto maybeAllocOp = findAlloc(dest);
  if (failed(maybeAllocOp))
    return false;
  return (*maybeAllocOp).getResult() == buffer;
}

/// Utility function to update the subtree rooted in the rock.gpuAlloc()
/// As we go down the tree, we can meet
///    a) Transformations (rock.transform)
///    b) Operations that accept transformed views (e.g.,
///    threadwise_read_into/threadwise_write_all) c) Operations that accept raw
///    views (e.g., blockwise_gemm) d) memref.view to reinterpret the raw
///    allocation and pass the buffer to other operations
static void replaceUsesAndPropagateType(RewriterBase &rewriter, Location loc,
                                        Operation *oldOp, Value val,
                                        LoopLikeOpInterface loop,
                                        int64_t loopLength,
                                        int64_t multiBufferingFactor) {
  SmallVector<Operation *> opsToDelete;

  for (OpOperand &use : oldOp->getUses()) {
    OpBuilder::InsertionGuard g(rewriter);
    Operation *owner = use.getOwner();
    rewriter.setInsertionPoint(owner);
    // memref.view case: we expand the view type using the mbFactor. We call
    // this expansion a reinterpret_multibuffer operation, that we can use
    // to adjust the multibuffer factor at a later stage
    if (auto view = dyn_cast<memref::ViewOp>(owner)) {
      auto newView = rewriter.create<rock::ReinterpretMultiBufferOp>(
          loc, val, view.getType(), multiBufferingFactor);
      replaceUsesAndPropagateType(rewriter, loc, view, newView, loop,
                                  loopLength, multiBufferingFactor);
    } else if (auto transform = dyn_cast<TransformOp>(owner)) {
      // rock.transform case
      Value toTransform = val;
      if (isBottomOfTransformStack(val) && val.getType().isa<ShapedType>()) {
        // if this is the top of the transform stack (i.e., the input to the
        // transform is not a transform) add a broadcast
        auto shape = val.getType().cast<ShapedType>().getShape();
        BottomUpTMBuilder broadcast(rewriter, shape, loc);
        broadcast.broadcast({0}, {loopLength});
        for (unsigned int dimIdx = 1; dimIdx < shape.size(); dimIdx++) {
          broadcast.passThrough({dimIdx}, {dimIdx});
        }
        TransformMapAttr broadcastAttr = broadcast.get();
        toTransform = rewriter.create<TransformOp>(loc, val, broadcastAttr);
      }
      // Change the transform stack adding a passthrough for the multibuffer
      // index
      auto newMap = getMultiBufferTransform(
          oldOp->getContext(), transform.getTransformAttr(), loopLength);
      auto newTransform =
          rewriter.create<TransformOp>(loc, toTransform, newMap);

      replaceUsesAndPropagateType(rewriter, loc, transform, newTransform, loop,
                                  loopLength, multiBufferingFactor);
    } else if (acceptsViewAt(owner, use)) {
      // This is an operation that accepts a view at the current position:
      // Add the loop index as an additional index
      auto viewAcceptingOp = dyn_cast<RockAcceptingViewOpInterface>(owner);
      SmallVector<Value> newExtraIndices = {*loop.getSingleInductionVar()};
      auto extraIndices = viewAcceptingOp.getExtraIndices(use);
      if (extraIndices)
        llvm::append_range(newExtraIndices, *extraIndices);
      Operation *newOp = viewAcceptingOp.cloneWithExtraIndices(
          rewriter, use, val, newExtraIndices);
      if (newOp->getNumResults())
        rewriter.replaceAllUsesWith(owner->getResults(), newOp->getResults());
    } else {
      // This is an operation that does not accept a view at the current
      // position We need to add a straight affine map to do ( %i mod mbFactor)
      Value mbIndex =
          getMultiBufferIndex(rewriter, loc, loop, multiBufferingFactor);
      // And then we can subview into the multibuffer
      Operation *newOp = subviewAndClone(rewriter, loc, use, val, mbIndex,
                                         multiBufferingFactor);
      if (newOp->getNumResults())
        rewriter.replaceAllUsesWith(owner->getResults(), newOp->getResults());
    }
    opsToDelete.push_back(owner);
  }
  // Delete the tree that is not used
  for (Operation *op : opsToDelete) {
    if (op->getUses().empty())
      rewriter.eraseOp(op);
  }
}
} // namespace

/// Transformation to do multi-buffering/array expansion to remove dependencies
/// on the temporary allocation between consecutive loop iterations.
/// Returns success if the transformation happened and failure otherwise.
/// This is not a pattern as it requires propagating the new memref type to its
/// uses and requires updating subview ops.
///
/// The skeleton of the function driver is insipired to
///   mlir/lib/Dialect/MemRef/Transforms/MultiBuffer.cpp
///
/// We mostly changed the recursive update function to adapt to our threadwise
/// operations
FailureOr<rock::GpuAllocOp>
mlir::rock::multiBuffer(RewriterBase &rewriter, rock::GpuAllocOp allocOp,
                        unsigned multiBufferingFactor,
                        bool skipOverrideAnalysis) {
  LLVM_DEBUG(DBGS() << "Start multibuffering: " << allocOp << "\n");
  if (!allocOp.getType().getElementType().isInteger(8)) {
    LLVM_DEBUG(DBGS() << "-- Not a int8 buffer -> fail\n");
    return failure();
  }
  if (allocOp.getType().getShape().size() != 1) {
    LLVM_DEBUG(DBGS() << "-- Not a flat buffer -> fail\n");
    return failure();
  }

  bool isUsedByViews = llvm::all_of(allocOp->getUsers(), [](Operation *user) {
    return dyn_cast<memref::ViewOp>(user);
  });
  if (!isUsedByViews) {
    LLVM_DEBUG(DBGS() << "-- Cannot detect the raw i8 buffer alloc followed by "
                         "a memref.viewOp\n");
    return failure();
  }

  DominanceInfo dom(allocOp->getParentOp());
  LoopLikeOpInterface candidateLoop;
  SmallPtrSet<Operation *, 16> users;
  findAllocUsers(allocOp, users);

  for (Operation *user : users) {
    auto parentLoop = user->getParentOfType<LoopLikeOpInterface>();
    if (!parentLoop) {
      LLVM_DEBUG(DBGS() << "--no parent loop -> fail\n");
      LLVM_DEBUG(DBGS() << "----due to user: " << *user << "\n");
      return failure();
    }
    if (!skipOverrideAnalysis) {
      /// Make sure there is no loop-carried dependency on the allocation.
      if (!overrideBuffer(user, allocOp.getResult())) {
        LLVM_DEBUG(DBGS() << "--Skip user: found loop-carried dependence\n");
        continue;
      }
      // If this user doesn't dominate all the other users keep looking.
      if (llvm::any_of(users, [&](Operation *otherUser) {
            return !dom.dominates(user, otherUser);
          })) {
        LLVM_DEBUG(
            DBGS() << "--Skip user: does not dominate all other users\n");
        continue;
      }
    } else {
      if (llvm::any_of(users, [&](Operation *otherUser) {
            return !isa<memref::DeallocOp>(otherUser) &&
                   !parentLoop->isProperAncestor(otherUser);
          })) {
        LLVM_DEBUG(
            DBGS()
            << "--Skip user: not all other users are in the parent loop\n");
        continue;
      }
    }
    candidateLoop = parentLoop;
    break;
  }

  if (!candidateLoop) {
    LLVM_DEBUG(DBGS() << "Skip alloc: no candidate loop\n");
    return failure();
  }

  std::optional<Value> inductionVar = candidateLoop.getSingleInductionVar();
  std::optional<OpFoldResult> lowerBound = candidateLoop.getSingleLowerBound();
  std::optional<OpFoldResult> upperBound = candidateLoop.getSingleUpperBound();
  std::optional<OpFoldResult> singleStep = candidateLoop.getSingleStep();
  if (!inductionVar || !lowerBound || !singleStep || !upperBound) {
    LLVM_DEBUG(DBGS() << "Skip alloc: no single iv, lb or step\n");
    return failure();
  }

  if (!dom.dominates(allocOp.getOperation(), candidateLoop)) {
    LLVM_DEBUG(DBGS() << "Skip alloc: does not dominate candidate loop\n");
    return failure();
  }

  LLVM_DEBUG(DBGS() << "Start multibuffering loop: " << candidateLoop << "\n");

  // 1. Construct the multi-buffered memref type.
  LLVM_DEBUG(DBGS() << "--original type: " << allocOp.getType() << "\n");

  if (!allocOp.getType().getElementType().isInteger(8)) {
    // We only apply multibuffering on raw bytes allocs
    return failure();
  }
  if (allocOp.getType().getShape().size() > 1) {
    // We only apply multibuffering on raw bytes allocs
    return failure();
  }
  MemRefType mbMemRefType =
      getFlatMultiBufferType(allocOp.getType(), multiBufferingFactor);
  LLVM_DEBUG(DBGS() << "--multi-buffered type: " << mbMemRefType << "\n");

  // 2. Create the multi-buffered alloc.
  Location loc = allocOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(allocOp);
  auto mbAlloc = rewriter.create<rock::GpuAllocOp>(
      loc, mbMemRefType, ValueRange{}, allocOp->getAttrs());
  LLVM_DEBUG(DBGS() << "--multi-buffered alloc: " << mbAlloc << "\n");

  // 3. Gather the loop length
  Value lbVal = getValueOrCreateConstantIndexOp(rewriter, loc, *lowerBound);
  Value ubVal = getValueOrCreateConstantIndexOp(rewriter, loc, *upperBound);
  std::optional<int64_t> loopLength = computeConstDiff(lbVal, ubVal);

  if (!loopLength)
    return failure();

  // 4. RAUW with the particular slice, taking modular rotation into account.
  replaceUsesAndPropagateType(rewriter, loc, allocOp, mbAlloc, candidateLoop,
                              *loopLength, multiBufferingFactor);

  // 5. Finally, erase the old allocOp.
  rewriter.eraseOp(allocOp);

  return mbAlloc;
}

FailureOr<rock::GpuAllocOp>
mlir::rock::multiBuffer(rock::GpuAllocOp allocOp, unsigned multiBufferingFactor,
                        bool skipOverrideAnalysis) {
  IRRewriter rewriter(allocOp->getContext());
  return multiBuffer(rewriter, allocOp, multiBufferingFactor,
                     skipOverrideAnalysis);
}
