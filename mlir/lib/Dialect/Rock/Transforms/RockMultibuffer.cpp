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
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
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

/// This is replacing a generic operation op(buffer) with op(multibuffer)
Operation *clone(RewriterBase &rewriter, Location loc, OpOperand &operand,
                 Value multibuffer) {
  int64_t operandNumber = operand.getOperandNumber();
  Operation *op = operand.getOwner();
  auto operands = op->getOperands();
  operands[operandNumber] = multibuffer;
  Operation *newOp = rewriter.clone(*op);
  newOp->setOperand(operandNumber, multibuffer);
  return newOp;
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
///    b) memref.view to split in multiple buffers
///    c) Operations that read/write from/to the alloc
static void replaceUsesAndPropagateType(RewriterBase &rewriter, Location loc,
                                        Operation *oldOp, ArrayRef<Value> vals,
                                        LoopLikeOpInterface loop) {
  SmallVector<Operation *> opsToDelete;
  size_t multibufferFactor = vals.size();

  for (OpOperand &use : oldOp->getUses()) {
    OpBuilder::InsertionGuard g(rewriter);
    Operation *owner = use.getOwner();
    rewriter.setInsertionPoint(owner);
    // memref.view case: we expand the view type using the mbFactor. We call
    // this expansion a reinterpret_multibuffer operation, that we can use
    // to adjust the multibuffer factor at a later stage
    if (auto view = dyn_cast<memref::ViewOp>(owner)) {
      SmallVector<Value> newViews(multibufferFactor);
      for (size_t i = 0; i < multibufferFactor; i++) {
        Value zeroByteOffset =
            rewriter.create<arith::ConstantIndexOp>(loc, int64_t(0));
        newViews[i] = rewriter.create<memref::ViewOp>(
            loc, view.getType(), vals[i], zeroByteOffset,
            /*dynamic dim sizes=*/ValueRange{});
      }

      replaceUsesAndPropagateType(rewriter, loc, view, newViews, loop);
    } else if (auto transform = dyn_cast<TransformOp>(owner)) {
      // Do nothing for transform, we will deal with then at the end
      // (note: since transforms are still being used, we don't even erase
      // them)
      replaceUsesAndPropagateType(rewriter, loc, transform, vals, loop);
    } else if (auto extractMultiBuffer =
                   dyn_cast<rock::ExtractMultiBufferOp>(owner)) {
      SmallVector<Value> extendedBuffers;
      // Remove the current buffer
      for (auto buffer : extractMultiBuffer.getBuffers()) {
        if (buffer != use.get()) {
          extendedBuffers.push_back(buffer);
        }
      }

      // extend with more buffers (if any)
      for (auto buffer : vals)
        extendedBuffers.push_back(buffer);

      auto inductionVar = extractMultiBuffer.getSelectIndex();

      Value newExtractMultibuffer = rewriter.create<rock::ExtractMultiBufferOp>(
          loc, vals.back().getType(), extendedBuffers, inductionVar);
      rewriter.replaceAllUsesWith(owner->getResults(), newExtractMultibuffer);
    } else {
      auto inductionVar = loop.getSingleInductionVar();
      if (!inductionVar) {
        llvm_unreachable("Loop variables should already have been tested.");
      }

      // Now vals contain the un-transformed new values and the `use` will be
      // the transformed old values All we want to do is:
      // 1. extract a buffer from the new values
      Value buffer = rewriter.create<rock::ExtractMultiBufferOp>(
          loc, vals.back().getType(), vals, inductionVar.value());
      // 2. extract the transforms from the old value and apply those to the
      // buffer extracted
      ArrayAttr transforms;
      std::tie(std::ignore, transforms, std::ignore) =
          untransform(rewriter, use.get());
      auto transformed = rock::transform(rewriter, buffer, transforms);
      // 3. delete the old transform ops
      Value ret = use.get();
      while (auto transform =
                 dyn_cast_or_null<TransformOp>(ret.getDefiningOp())) {
        opsToDelete.push_back(transform);
        ret = transform.getInput();
      }

      // And now we can use the buffer
      Operation *newOp = clone(rewriter, loc, use, transformed);
      if (newOp->getNumResults())
        rewriter.replaceAllUsesWith(owner->getResults(), newOp->getResults());
    }
    opsToDelete.push_back(owner);
  }
  // Delete the tree that is not used
  for (Operation *op : opsToDelete) {
    if (op->getUses().empty()) {
      rewriter.eraseOp(op);
    }
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
FailureOr<SmallVector<rock::GpuAllocOp>>
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

  // 2. Create the multiple buffers alloc.
  Location loc = allocOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(allocOp);
  SmallVector<rock::GpuAllocOp> mbAlloc(multiBufferingFactor);
  for (unsigned i = 0; i < multiBufferingFactor; i++) {
    mbAlloc[i] = rewriter.create<rock::GpuAllocOp>(
        loc, allocOp.getType(), ValueRange{}, allocOp->getAttrs());
  }

  // 3. RAUW with the particular slice, taking modular rotation into account.

  SmallVector<Value> startingValues;
  for (auto alloc : mbAlloc) {
    startingValues.push_back(alloc);
  }
  replaceUsesAndPropagateType(rewriter, loc, allocOp, startingValues,
                              candidateLoop);

  // 4. Finally, erase the old allocOp.
  rewriter.eraseOp(allocOp);

  return mbAlloc;
}

FailureOr<SmallVector<rock::GpuAllocOp>>
mlir::rock::multiBuffer(rock::GpuAllocOp allocOp, unsigned multiBufferingFactor,
                        bool skipOverrideAnalysis) {
  IRRewriter rewriter(allocOp->getContext());
  return multiBuffer(rewriter, allocOp, multiBufferingFactor,
                     skipOverrideAnalysis);
}

FailureOr<SmallVector<rock::GpuAllocOp>>
mlir::rock::updateMultiBuffer(RewriterBase &rewriter, Location loc,
                              ArrayRef<rock::GpuAllocOp> allocs,
                              unsigned newMultiBufferingFactor) {
  // Select the first alloc, which will be updated
  rock::GpuAllocOp allocOp = allocs.back();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(allocOp);
  SmallVector<GpuAllocOp> newAllocs;
  if (allocs.size() < newMultiBufferingFactor) {
    for (unsigned i = 0; i < newMultiBufferingFactor - allocs.size() + 1; i++)
      newAllocs.push_back(rewriter.create<rock::GpuAllocOp>(
          loc, allocOp.getType(), ValueRange{}, allocOp->getAttrs()));
    SmallVector<Value> startingValues;
    for (auto alloc : newAllocs) {
      startingValues.push_back(alloc);
    }
    replaceUsesAndPropagateType(rewriter, loc, allocOp, startingValues,
                                nullptr);
  } else if (allocs.size() > newMultiBufferingFactor) {
    for (unsigned i = 0; i < allocs.size() - newMultiBufferingFactor; i++)
      replaceUsesAndPropagateType(rewriter, loc, allocOp, {}, nullptr);
  }
  return newAllocs;
}

FailureOr<SmallVector<rock::GpuAllocOp>>
mlir::rock::updateMultiBuffer(ArrayRef<rock::GpuAllocOp> allocs,
                              unsigned newMultiBufferingFactor) {
  // Select the first alloc, which will be updated
  rock::GpuAllocOp allocOp = allocs.back();
  IRRewriter rewriter(allocOp->getContext());
  Location loc = allocOp.getLoc();
  return updateMultiBuffer(rewriter, loc, allocs, newMultiBufferingFactor);
}
