//===- AlignTiling.cpp - Align Linalg ops with Rock ops -------------------===//
//
// Copyright 2020 The MLIR Authors.
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
// =============================================================================
//
// This pass refactors linalg.generic ops from global scope to tiled scope
// based on rock lowering step2.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"

#include <deque>
#include <numeric>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLINALGALIGNPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-linalg-align"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockLinalgAlignPass
    : public rock::impl::RockLinalgAlignPassBase<RockLinalgAlignPass> {
  void runOnOperation() override;
};

/// The patterns in this file do not and cannot fulfill the contract of
/// upstream's greedy pattern rewriter. Therefore, we implement the following
/// custom rewriter, which:
/// - Allows explicitly inserting operations into the worklist, to allow
///   recursively scheduling rewrites on linalg.generic blocks that need
///   tiling information propagated up.
/// - Doesn't needlessly re-schedule the fusion pattern onto the linalg.generic
///   that has already been modified.
/// - Doesn't spend time peeking into regions
/// - Propagates rewrite failure to the main code. However, if a pattern
///   fails but the op indicates that it expects to be visited in a future pass,
///   this is recorded, and the failure only counts if the re-visit doesn't
///   happen.
/// - Assumes all patterns match some specific operation to save time
///   constructing the initial worklist.
struct LinalgAlignRewriter : public PatternRewriter,
                             public RewriterBase::Listener {
protected:
#ifndef NDEBUG
  llvm::ScopedPrinter logger{llvm::dbgs()};
#endif

  std::deque<Operation *> worklist;
  // Our patterns aren't stable under repeated application (ex. fuling an
  // already fused op) but we can have cases where an operation is marked for
  // multiple visits (ex. two arguments to a linalg.generic reading from
  // the same source linalg.generic), so we need to prevent duplicate
  // scheduling.
  llvm::SmallPtrSet<Operation *, 4> scheduledOps;

  SmallPtrSet<Operation *, 2> opsExpectingRevisit;

  llvm::SmallDenseSet<OperationName> matchableOps;
  bool hasMatchingPattern(Operation *op);

  /// Utility for debug logging.
  void logOpActivity(llvm::StringLiteral prefix, Operation *op);

public:
  LinalgAlignRewriter(MLIRContext *ctx);

  /// Schedule the given operation for processing.
  void scheduleVisit(Operation *op);

  /// Inform the pattern rewriter that this operation will need to be revisited
  /// in the future to make everything correct. Returns success() (we matched,
  /// but have to wait to actually run this) as a convenience. If the operation
  /// is never revisited, then fail.
  LogicalResult needsRevisit(Operation *op);

  /// Match all worklist itemis against the patterns in `matcher`, repeating
  /// this procedure until all scheduled visits are complete, including those
  /// added during a pattern match. Fails if any pattern application fails.
  /// or if an operation that is marked as needing revisiting is not visited
  /// after it is so marked.
  LogicalResult drainWorklist(PatternApplicator &matcher);

  /// Move `toMove` after `earliestOp` if `toMove` is before `earliestOp`
  /// in the IR.
  void moveAfterIfNeeded(Operation *toMove, Operation *earliestOp);

  /// Move `toMove` before `latestPoint` if `toMove` is after `latestOp` in
  /// the IR.
  void moveBeforeIfNeeded(Operation *toMove, Operation *latestOp);

  /// Debug utilities.
  void notifyOperationModified(Operation *op) override;
  void notifyOperationInserted(Operation *op, InsertPoint previous) override;
  void notifyOperationErased(Operation *op) override;
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override;
  void notifyBlockInserted(mlir::Block *block, mlir::Region *previous,
                                   mlir::Region::iterator previousIt) override;
  using PatternRewriter::notifyMatchFailure;
  void
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback) override;
};

template <typename Op>
struct AlignRewritePattern : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  using OpRewritePattern<Op>::matchAndRewrite;
  virtual LogicalResult matchAndRewrite(Op op,
                                        LinalgAlignRewriter &b) const = 0;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    LinalgAlignRewriter &b = static_cast<LinalgAlignRewriter &>(rewriter);
    return this->matchAndRewrite(op, b);
  }
};

struct LAGenericRewritePattern : public AlignRewritePattern<linalg::GenericOp> {
  using AlignRewritePattern<linalg::GenericOp>::AlignRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp laGeneric,
                                LinalgAlignRewriter &b) const override;
};

struct MemcpyRewritePattern : public AlignRewritePattern<memref::CopyOp> {
  using AlignRewritePattern<memref::CopyOp>::AlignRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                LinalgAlignRewriter &b) const override;
};

struct ReduceRewritePattern : public AlignRewritePattern<rock::ReduceOp> {
  using AlignRewritePattern<rock::ReduceOp>::AlignRewritePattern;

  LogicalResult matchAndRewrite(rock::ReduceOp reduceOp,
                                LinalgAlignRewriter &rewriter) const override;
};
} // end anonymous namespace

/// Pattern rewriter
LinalgAlignRewriter::LinalgAlignRewriter(MLIRContext *ctx)
    : PatternRewriter(ctx) {
  setListener(this);
}

bool LinalgAlignRewriter::hasMatchingPattern(Operation *op) {
  return matchableOps.contains(op->getName());
}

void LinalgAlignRewriter::moveAfterIfNeeded(Operation *toMove,
                                            Operation *earliestOp) {
  constexpr llvm::StringLiteral movePrefix("** Move    : ");
  constexpr llvm::StringLiteral afterPrefix("   after   : ");
  logOpActivity(movePrefix, toMove);
  logOpActivity(afterPrefix, earliestOp);
  if (toMove->isBeforeInBlock(earliestOp))
    toMove->moveAfter(earliestOp);
  else
    LLVM_DEBUG(logger.startLine() << "   No move needed.\n");
}

void LinalgAlignRewriter::moveBeforeIfNeeded(Operation *toMove,
                                             Operation *latestOp) {
  constexpr llvm::StringLiteral movePrefix("** Move    : ");
  constexpr llvm::StringLiteral beforePrefix("   before  : ");
  logOpActivity(movePrefix, toMove);
  logOpActivity(beforePrefix, latestOp);
  if (latestOp->getBlock() != toMove->getBlock() ||
      latestOp->isBeforeInBlock(toMove))
    toMove->moveBefore(latestOp);
  else
    LLVM_DEBUG(logger.startLine() << "   No move needed.\n");
}

void LinalgAlignRewriter::scheduleVisit(Operation *op) {
  constexpr llvm::StringLiteral prefix("** To visit: ");
  constexpr llvm::StringLiteral dupePrefix("** Duplicate:  ");
  if (scheduledOps.insert(op).second) {
    logOpActivity(prefix, op);
    worklist.push_back(op);
  } else {
    logOpActivity(dupePrefix, op);
  }
}

LogicalResult LinalgAlignRewriter::needsRevisit(Operation *op) {
  constexpr llvm::StringLiteral prefix("** Needs revisiting : ");
  logOpActivity(prefix, op);
  opsExpectingRevisit.insert(op);
  return success();
}

LogicalResult LinalgAlignRewriter::drainWorklist(PatternApplicator &matcher) {
  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop_front();
#ifndef NDEBUG
    auto canApply = [&](const Pattern &pattern) -> bool {
      LLVM_DEBUG({
        logger.startLine() << "Applying pattern " << pattern.getDebugName()
                           << " (matches" << *pattern.getRootKind() << ")"
                           << " on " << op->getName() << "\n";
        logger.indent();
      });
      return true;
    };
    auto onFailure = [&](const Pattern &pattern) {
      LLVM_DEBUG({
        logger.unindent();
        logger.startLine() << "Failed to match the " << pattern.getDebugName()
                           << " pattern (matches " << *pattern.getRootKind()
                           << ") on " << op->getName() << "\n";
      });
    };
    auto onSuccess = [&](const Pattern &pattern) -> LogicalResult {
      LLVM_DEBUG({
        logger.unindent();
        logger.startLine() << "Matched " << pattern.getDebugName()
                           << " pattern on " << op->getName() << "\n";
      });
      return success();
    };
#else
    function_ref<bool(const Pattern &)> canApply = {};
    function_ref<void(const Pattern &)> onFailure = {};
    function_ref<LogicalResult(const Pattern &)> onSuccess = {};
#endif
    scheduledOps.erase(op);
    opsExpectingRevisit.erase(op);
    LogicalResult matchResult =
        matcher.matchAndRewrite(op, *this, canApply, onFailure, onSuccess);
    if (failed(matchResult)) {
      LLVM_DEBUG(llvm::dbgs() << "Pattern match failed\n");
      return failure();
    }
  }
  if (!opsExpectingRevisit.empty()) {
#ifndef NDEBUG
    for (Operation *op : opsExpectingRevisit) {
      LLVM_DEBUG(logger.startLine() << "Failed to revisit " << *op << "\n");
    }
#endif
    return failure();
  }
  return success();
}

LogicalResult applyAlignPatterns(Operation *op,
                                 const FrozenRewritePatternSet &patterns) {
  LinalgAlignRewriter rewriter(op->getContext());
  PatternApplicator matcher(patterns);

  llvm::SmallDenseSet<OperationName, 4> matchableOps;
  for (const auto &entry : patterns.getOpSpecificNativePatterns())
    matchableOps.insert(entry.first);
  assert(patterns.getMatchAnyOpNativePatterns().empty() &&
         "We're assuming no catch-all logic");
  matcher.applyDefaultCostModel();

  // Initialize worklist with patterns that might need to be processed
  // top-down.
  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        if (matchableOps.contains(op.getName())) {
          rewriter.scheduleVisit(&op);
        }
      }
    }
  }

  return rewriter.drainWorklist(matcher);
}

void LinalgAlignRewriter::logOpActivity(llvm::StringLiteral prefix,
                                        Operation *op) {
#ifndef NDEBUG
  LLVM_DEBUG({
    logger.startLine() << prefix;
    op->print(logger.getOStream(), OpPrintingFlags()
                                       .skipRegions()
                                       .printGenericOpForm()
                                       .useLocalScope()
                                       .elideLargeElementsAttrs());
    logger.getOStream() << "\n";
  });
#else
  std::ignore = prefix;
  std::ignore = op;
#endif
}

void LinalgAlignRewriter::notifyOperationModified(Operation *op) {
  constexpr llvm::StringLiteral prefix("** Modified: ");
  logOpActivity(prefix, op);
}

void LinalgAlignRewriter::notifyOperationInserted(Operation *op, InsertPoint previous) {
  assert(!previous.isSet() && "expected newly created op");
  constexpr llvm::StringLiteral prefix("** Insert  : '");
  logOpActivity(prefix, op);
}

void LinalgAlignRewriter::notifyOperationErased(Operation *op) {
  constexpr llvm::StringLiteral prefix("** Erase   : ");
  logOpActivity(prefix, op);
}

void LinalgAlignRewriter::notifyOperationReplaced(Operation *op,
                                                  ValueRange replacement) {
  constexpr llvm::StringLiteral prefix("** Replace : ");
  logOpActivity(prefix, op);
  std::ignore = replacement;
}

void LinalgAlignRewriter::notifyBlockInserted(mlir::Block *block, mlir::Region *previous,
                                  mlir::Region::iterator previousIt) {
  std::ignore = block;
}

void LinalgAlignRewriter::notifyMatchFailure(
    Location loc, function_ref<void(Diagnostic &)> reasonCallback) {
  LLVM_DEBUG({
    Diagnostic diag(loc, DiagnosticSeverity::Remark);
    reasonCallback(diag);
    logger.startLine() << "** Failure : " << diag.str() << "\n";
  });
#ifdef NDEBUG
  std::ignore = loc;
  std::ignore reasonCallback;
#endif
}

/// Fusion

static Value applyViewsOnDest(LinalgAlignRewriter &rewriter, Location loc,
                              Value dest, ArrayRef<TransformMapAttr> views) {
  for (TransformMapAttr trMap : llvm::reverse(views)) {
    dest = rewriter.create<TransformOp>(loc, dest, trMap);
  }
  return dest;
}

// This function traces to non view readers.
// TODO(@manupak): we should really be implementing
// MemoryEffectOpInterface to ops to make this simpler
// to identify readers.
static LogicalResult
traceToNonViewReaders(Operation *op, Value parentVal,
                      SmallVectorImpl<Operation *> &nonViewReaders) {
  if (auto transform = dyn_cast<TransformOp>(op)) {
    Value result = transform.getResult();
    bool recurStatus = true;
    for (auto &use : result.getUses()) {
      recurStatus &= succeeded(
          traceToNonViewReaders(use.getOwner(), result, nonViewReaders));
    }
    return success(recurStatus);
  }
  if (auto twWriteAllOp = dyn_cast<ThreadwiseWriteAllOp>(op)) {
    if (twWriteAllOp.getSource() == parentVal) {
      nonViewReaders.push_back(op);
    }
    // linalg.generic is conservative when it comes to memory effects, so it
    // needs to be handled separately.
  } else if (auto linalgGeneric = dyn_cast<linalg::GenericOp>(op)) {
    if (llvm::is_contained(linalgGeneric.getInputs(), parentVal)) {
      nonViewReaders.push_back(op);
    }
  } else if (auto memEffectOp = dyn_cast<MemoryEffectOpInterface>(op)) {
    if (hasEffect<MemoryEffects::Read>(memEffectOp, parentVal)) {
      nonViewReaders.push_back(op);
    }
  } else if (auto copyOp = dyn_cast<CopyOpInterface>(op)) {
    if (copyOp.getSource() == parentVal) {
      nonViewReaders.push_back(op);
    }
  } else {
    return op->emitError() << "Found an unsupported operator that needs to "
                              "be added reader checks \n"
                           << op;
  }
  return success();
}

// This function checks given a parent operation and a reader candidate
// whether that reader is the unique reader of the parent operation.
// This is to be used identify where interrim memrefs could be eliminated
// and fused.
static LogicalResult checkUniqueReader(Operation *op, Operation *reader,
                                       bool &isUnique) {
  while (auto trOp = dyn_cast_if_present<TransformOp>(op))
    op = trOp.getViewSource().getDefiningOp();
  SmallVector<Operation *> nonViewReaders;
  for (Value result : op->getResults()) {
    // if its block arg, it can have uses beyond the unit of compilation
    // in scope here.
    if (result.isa<BlockArgument>()) {
      isUnique = false;
    }
    for (auto &use : result.getUses()) {
      LogicalResult traceResult =
          traceToNonViewReaders(use.getOwner(), result, nonViewReaders);
      if (traceResult.failed()) {
        return traceResult;
      }
    }
  }
#ifndef NDEBUG
  LLVM_DEBUG(llvm::dbgs() << "Non-view readers:\n");
  for (Operation *reader : nonViewReaders)
    LLVM_DEBUG(llvm::dbgs() << *reader << "\n");
#endif
  if (nonViewReaders.size() != 1)
    isUnique = false;
  else
    isUnique = nonViewReaders[0] == reader;
  return success();
}

static Operation *
traceToWriter(Value startVal,
              SmallVectorImpl<TransformMapAttr> &writerToStartValViews) {
  // 1. Validate that the only uses of the linalg.generic input are the one
  // generic and a copy operation or transform.
  Operation *result = nullptr;
  Operation *startValDef = startVal.getDefiningOp();
  if (!startValDef)
    return nullptr;
  bool unique = true;
  auto setResult = [&](Value val, Operation *theResult) {
    if (result != nullptr) {
      LLVM_DEBUG(llvm::dbgs() << "Found writer " << *theResult
                              << " after finding " << *result << "\n");
      unique = false;
    } else {
      result = theResult;
      TransformOp trOp = dyn_cast_if_present<TransformOp>(val.getDefiningOp());
      while (trOp && trOp != startValDef) {
        writerToStartValViews.push_back(trOp.getTransformAttr());
        trOp = dyn_cast_if_present<TransformOp>(
            trOp.getViewSource().getDefiningOp());
      }
      // This recursion got us our transform stack in the opposite order.
      llvm::reverse(writerToStartValViews);
    }
  };
  SmallVector<std::pair<Value, Operation *>> worklist =
      llvm::map_to_vector(startVal.getUsers(), [&](Operation *op) {
        return std::make_pair(startVal, op);
      });
  while (!worklist.empty()) {
    auto [val, use] = worklist.pop_back_val();
    if (auto transformOp = dyn_cast<TransformOp>(use)) {
      for (Operation *transformUse : transformOp->getUsers()) {
        worklist.push_back({transformOp, transformUse});
      }
    } else if (auto lgop = dyn_cast<linalg::GenericOp>(use)) {
      if (llvm::is_contained(lgop.getOutputs(), val))
        setResult(val, use);
    } else if (auto reduceOp = dyn_cast<ReduceOp>(use)) {
      if (val == reduceOp.getOut())
        setResult(val, use);
    } else if (auto memcpy = dyn_cast<memref::CopyOp>(use)) {
      if (memcpy.getTarget() == val)
        setResult(val, use);
    } else if (auto store = dyn_cast<ThreadwiseWriteAllOp>(use)) {
      if (store.getDest() == val)
        setResult(val, use);
    }
  }

  if (!unique) {
    LLVM_DEBUG(llvm::dbgs() << "Found multiple writers somehow\n");
    result = nullptr;
  }
  return result;
}

static Value makeRegs(LinalgAlignRewriter &b, MemRefType::Builder &mrb,
                      Location loc, Type srcType) {
  auto srcMemType = srcType.cast<MemRefType>();
  // 1. create a second allocation of the same type to hold loaded elements
  return b.create<GpuAllocOp>(loc, static_cast<MemRefType>(mrb.setElementType(
                                       srcMemType.getElementType())));
}

static void markGenericWritersToRevisit(LinalgAlignRewriter &b, Value rawSrc) {
  SmallVector<TransformMapAttr> views;
  auto genericWriter =
      dyn_cast_if_present<linalg::GenericOp>(traceToWriter(rawSrc, views));
  if (genericWriter)
    b.scheduleVisit(genericWriter);
}

template <typename TiledOp> Value getRegisterValue(TiledOp op);
template <>
Value getRegisterValue<ThreadwiseReadIntoOp>(ThreadwiseReadIntoOp op) {
  return op.getDest();
}
template <>
Value getRegisterValue<ThreadwiseWriteAllOp>(ThreadwiseWriteAllOp op) {
  return op.getSource();
}

/// Given a `tiledOp` that has as its arguments the thread tile (a set of
/// registers) and a global buffer (with some series of views applied, which
/// have been collected into `globalCoordsToGenericViews`) indexed by a set of
/// global coordinates, read the equivalent tile of `src` into a newly-created
/// set of registers. The views in `globalCoordsToGenericViews` should produce
/// coordinates that can index `src` (which will have the same size as the other
/// generic inputs at this point thanks to regularization processes and view
/// application).
///
/// While doing this, also mark any linalg.generic that write to this input as
/// needing revisiting because we now know their tile size.
///
/// Returns the new register tile.
template <typename TiledOp>
static Value
makeExtraInputTile(LinalgAlignRewriter &b, TiledOp tiledOp, Value src,
                   ArrayRef<TransformMapAttr> globalCoordsToGenericViews,
                   linalg::GenericOp laGeneric) {
  // 0. capture the memref containing the outputs being written or
  // (in the case of propagating tiling informatinon up to gemm-independent
  // code) where the values will be written.
  Location loc = tiledOp.getLoc();
  Value tile = getRegisterValue(tiledOp);

  // 1. create a second allocation of the same type to hold loaded elements
  MemRefType::Builder mrb(tile.getType().cast<MemRefType>());
  Value alloc = makeRegs(b, mrb, loc, src.getType());

  // 1.1. Find out if the source is a scalar so we don't unroll a memset()
  Value rawSrc = std::get<0>(untransform(b, src));
  bool forceUnroll = tiledOp.getForceUnroll();
  bool useIndexDiffs = tiledOp.getUseIndexDiffs();
  auto baseOpType = cast<MemRefType>(rawSrc.getType());
  if (baseOpType.getNumElements() == 1) {
    forceUnroll = false;
    useIndexDiffs = false;
  }

  // 2. clone twcopy for <addend> into regs
  LLVM_DEBUG(llvm::dbgs() << "Src type: " << src.getType()
                          << " tile type: " << tile.getType() << "\n");

  // 2.0. apply transform chain from output
  src = applyViewsOnDest(b, loc, src, globalCoordsToGenericViews);

  // 2.1. move linalg.generic if needed
  // move linalg.generic after the definitions of threadwiseReadIntoOp's
  // inputs to maintaine correct def-use chain.
  Operation *lastIdxDef = nullptr;
  for (Value idx : tiledOp.getExtraIndices()) {
    Operation *idxOp = idx.getDefiningOp();
    if (idxOp) {
      if (!lastIdxDef || lastIdxDef->isBeforeInBlock(idxOp)) {
        lastIdxDef = idxOp;
      }
    }
  }
  if (lastIdxDef)
    b.moveAfterIfNeeded(laGeneric, lastIdxDef);

  // move linalg.generic in the same block of threadwiseReadIntoOp
  if (laGeneric->getBlock() != tiledOp->getBlock()) {
    b.moveBeforeIfNeeded(laGeneric, tiledOp);
    b.setInsertionPoint(laGeneric);
  }

  // 2.2. load into registers
  ThreadwiseReadIntoOp threadwiseReadIntoOp = b.create<ThreadwiseReadIntoOp>(
      loc, src, alloc, tiledOp.getExtraViews(),
      /*extraIndices=*/tiledOp.getExtraIndices(), forceUnroll, useIndexDiffs);

  // 3. Mark linalg.generic operations that populate this source buffer as
  // operations that need to be re-checekd for fusion now that we know their
  // tiling.
  markGenericWritersToRevisit(b, rawSrc);

  return alloc;
}

static Value findThreadwiseWrite(
    linalg::GenericOp laGeneric, ThreadwiseWriteAllOp &twWriteOp,
    SmallVectorImpl<TransformMapAttr> &globalCoordsToGenericViews) {
  for (auto input : laGeneric.getInputs()) {
    if (auto twop = dyn_cast_if_present<ThreadwiseWriteAllOp>(
            traceToWriter(input, globalCoordsToGenericViews))) {
      twWriteOp = twop;
      return input;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "No input is leading to a global store.\n");
  return Value();
}

/// Find, if there is one, the threadwise_read_into that reads the output of
/// this `linalg.generic` so that we have a tile size to use. If this fails,
/// we'll neeed to revisit the operation later once we have a tile size.
///
/// Returns failure on an error during the search, and fails to set twReadOp
/// if no reader is found.
static LogicalResult findThreadwiseRead(
    linalg::GenericOp laGeneric, ThreadwiseReadIntoOp &twReadOp,
    SmallVectorImpl<TransformMapAttr> &globalCoordsToGenericViews) {
  Value out = laGeneric.getOutputs().front();
  SmallVector<Operation *> readers;
  for (Operation *user : out.getUsers()) {
    if (user == laGeneric)
      continue;
    if (failed(traceToNonViewReaders(user, out, readers)))
      return failure();
  }
  for (Operation *reader : readers) {
    twReadOp = dyn_cast<ThreadwiseReadIntoOp>(reader);
    if (twReadOp) {
      auto [underlying, isBig] =
          untransform(twReadOp.getSource(), globalCoordsToGenericViews);
      std::ignore = isBig;
      if (underlying != out) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "A non-view threadwise_read_into of this linalg.generic's "
               "output doesn't trace back to said output\n");
        return failure();
      }
      return success();
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "Output doesn't lead to a tiled read either\n");
  return success();
}

// Rewrite the inputs of a given linalg.generic to match the tiling from
// the given `twWriteOp`, repalacing each input other than the one that the
// `twWriteOp` was writing to with a `threadwise_read_into` with the
// appropriate registers.
static void addRegisterReadsForTiledInput(
    LinalgAlignRewriter &b, linalg::GenericOp laGeneric,
    Value laGenericInputLeadingToWrite, ThreadwiseWriteAllOp twWriteOp,
    ArrayRef<TransformMapAttr> relativeViewsOnWrite,
    SmallVectorImpl<Value> &newInputs) {
  for (auto inp : laGeneric.getInputs()) {
    Value newInput;
    if (inp == laGenericInputLeadingToWrite) {
      newInput = twWriteOp.getSource();
    } else {
      newInput = makeExtraInputTile(b, twWriteOp, inp, relativeViewsOnWrite,
                                    laGeneric);
    }
    newInputs.push_back(newInput);
  }
}

/// As above, but applying
static void
addRegisterReadsForTiledOutput(LinalgAlignRewriter &b,
                               linalg::GenericOp laGeneric,
                               ThreadwiseReadIntoOp twReadOp,
                               ArrayRef<TransformMapAttr> relativeViewsOnResult,
                               SmallVectorImpl<Value> &newInputs) {
  for (auto inp : laGeneric.getInputs()) {
    Value newInput =
        makeExtraInputTile(b, twReadOp, inp, relativeViewsOnResult, laGeneric);
    newInputs.push_back(newInput);
  }
}

static void reconfigureLAGeneric(LinalgAlignRewriter &b,
                                 linalg::GenericOp laGeneric,
                                 ValueRange newInputs, Value newOutput) {
  SmallVector<AffineMap, 5> lgAMaps;

  for (Value newInput : newInputs) {
    auto inpRank = newInput.getType().cast<ShapedType>().getRank();
    lgAMaps.push_back(b.getMultiDimIdentityMap(inpRank));
  }

  laGeneric.getInputsMutable().assign(newInputs);
  laGeneric.getOutputsMutable().assign(newOutput);

  auto regRank = newOutput.getType().cast<ShapedType>().getRank();

  lgAMaps.push_back(b.getMultiDimIdentityMap(regRank));
  laGeneric.setIndexingMapsAttr(b.getAffineMapArrayAttr(lgAMaps));

  // 2.3. Reset iterator types
  MLIRContext *ctx = b.getContext();
  SmallVector<Attribute, 5> iteratorTypes;
  iteratorTypes.resize(regRank, linalg::IteratorTypeAttr::get(
                                    ctx, utils::IteratorType::parallel));
  laGeneric.setIteratorTypesAttr(ArrayAttr::get(ctx, iteratorTypes));
}

LogicalResult
LAGenericRewritePattern::matchAndRewrite(linalg::GenericOp laGeneric,
                                         LinalgAlignRewriter &b) const {
  Location loc = laGeneric.getLoc();

  // 0. Test compatibility
  // 0.0. Only fully parallel for now
  for (utils::IteratorType iterType : laGeneric.getIteratorTypesArray())
    if (!linalg::isParallelIterator(iterType))
      return laGeneric.emitOpError("must be fully parallel");

  // 0.1. Test compatibility,  Only 1 output supported
  if (laGeneric.getOutputs().size() != 1)
    return laGeneric.emitOpError("only 1 output supported");
  Value out = *laGeneric.getOutputs().begin();

  // 0.2. Prevent re-applying pattern to existing fusions.
  if (out.getDefiningOp<rock::GpuAllocOp>())
    return b.notifyMatchFailure(loc,
                                "encountered already-processed fusion op\n");

  for (auto idxMap : laGeneric.getIndexingMapsArray())
    if (!idxMap.isIdentity())
      return b.notifyMatchFailure(loc, "indexing_maps must all be identity");

  // 1. Find the tiling needed for this linalg generic.
  // 1.1. Find the (implicit) gemm output, if it exists.
  ThreadwiseWriteAllOp gemmStoreOp;
  SmallVector<TransformMapAttr> globalCoordsToGenericViews;
  Value laGenericArgLeadingToTile =
      findThreadwiseWrite(laGeneric, gemmStoreOp, globalCoordsToGenericViews);

  // 1.2. If there is no input being written, try to find a threadwise_read_into
  // operation that reads from the output of this generic. If there is such an
  ThreadwiseReadIntoOp tileReadOp;
  if (!gemmStoreOp) {
    if (failed(findThreadwiseRead(laGeneric, tileReadOp,
                                  globalCoordsToGenericViews)))
      return b.notifyMatchFailure(
          loc, "search for readers hit something unexpected\n");
    if (!tileReadOp)
      // 1.2.1 If there's no threadwise reader, we don't know the tile size for
      // this generic yet, and so have to wait for it to by put on the schedule
      // by someone else using the input.
      return b.needsRevisit(laGeneric);
    laGenericArgLeadingToTile = out;
  }

  // 2. The less common case, where the tile size and global coordinates come
  // from a threadwise_read_into of the output of this generic.
  if (!gemmStoreOp && tileReadOp) {
    bool isUniqueReader = false;
    if (failed(checkUniqueReader(laGenericArgLeadingToTile.getDefiningOp(),
                                 tileReadOp, isUniqueReader)))
      return b.notifyMatchFailure(
          loc, "failed to check reader uniqueness of generic reader output");
    if (!isUniqueReader) {
      // Clone ourselves so that we can rewrite this writer without disturbing
      // the rest of the fusion process.
      b.scheduleVisit(laGeneric);
      laGeneric = cast<linalg::GenericOp>(b.clone(*laGeneric));
      // We have to move the insertion point to avoid SSA issues.
      b.setInsertionPoint(laGeneric);
    }
    Value newOutput = tileReadOp.getDest();
    SmallVector<Value> newInputs;
    addRegisterReadsForTiledOutput(b, laGeneric, tileReadOp,
                                   globalCoordsToGenericViews, newInputs);
    // Prevent SSA weirdness from register allocations introduced too late.
    b.moveBeforeIfNeeded(newOutput.getDefiningOp(), laGeneric);
    reconfigureLAGeneric(b, laGeneric, newInputs, newOutput);
    b.eraseOp(tileReadOp);
    return success();
  }
  auto outType = out.getType().cast<ShapedType>();
  auto inpType = laGenericArgLeadingToTile.getType().cast<ShapedType>();
  if (outType.getShape() != inpType.getShape()) {
    return laGeneric.emitError("input and output types must match");
  }

  // 3. The typical case, where there's in inputh that traced back to a
  // threadwise_write_all, which is the store of our gemm.
  bool isUniqueReader = false;
  if (failed(checkUniqueReader(laGenericArgLeadingToTile.getDefiningOp(),
                               laGeneric, isUniqueReader))) {
    LLVM_DEBUG(llvm::dbgs()
               << "This generic isn't the only reader from the gemm output\n");
  }
  if (!isUniqueReader) {
    gemmStoreOp = cast<ThreadwiseWriteAllOp>(b.clone(*gemmStoreOp));
  }

  Value gemmOutRegs = gemmStoreOp.getSource();
  auto gemmOutType = gemmOutRegs.getType().cast<MemRefType>();

  // 3.1. Make an allocation that matches the tile but has the type of the
  // linalg.generic output.
  MemRefType::Builder mrb(gemmOutType);
  Value laOutRegs = makeRegs(b, mrb, loc, out.getType());

  // 3.2. Tile linalg.generic with vgpr as input, return output vgprs
  SmallVector<Value> newInputs;
  addRegisterReadsForTiledInput(b, laGeneric, laGenericArgLeadingToTile,
                                gemmStoreOp, globalCoordsToGenericViews,
                                newInputs);
  reconfigureLAGeneric(b, laGeneric, newInputs, laOutRegs);

  // 4. Amend the tiled write to write the fusion result to the output of this
  // generic.

  // 4.1.  Prevent SSA issues from adjusting the write.
  b.moveAfterIfNeeded(gemmStoreOp, laGeneric);

  // 4.2. Replace rock.threadwise_write_all inputs with la.generic result vgprs
  gemmStoreOp.getSourceMutable().assign(laOutRegs);

  // 4.3 . Eliminate the intermediate allocation, applynig the chain of views
  // on top of the eliminated temporary to the generic's output.
  out = applyViewsOnDest(b, loc, out, globalCoordsToGenericViews);
  gemmStoreOp.getDestMutable().assign(out);
  return success();
}

LogicalResult
MemcpyRewritePattern::matchAndRewrite(memref::CopyOp copy,
                                      LinalgAlignRewriter &b) const {
  auto src = copy.getSource();
  auto target = copy.getTarget();
  Location loc = copy.getLoc();

  Operation *gemmStoreOp = nullptr;
  SmallVector<TransformMapAttr> views;
  if (auto allocOp = src.getDefiningOp<memref::AllocOp>()) {
    if (auto twop = dyn_cast_if_present<ThreadwiseWriteAllOp>(
            traceToWriter(src, views))) {
      // We check the input leading to GEMM store has the current memref
      // copy, that is being rewritten, as the unique reader. This is because if
      // it is the unique reader, the previous memref does not need to be
      // maintained anymore and we can directly write into the target of the
      // memref copy.
      bool isUniqueReader;
      LogicalResult checkResult =
          checkUniqueReader(src.getDefiningOp(), copy, isUniqueReader);
      if (checkResult.failed()) {
        return checkResult;
      }
      if (!isUniqueReader) {
        gemmStoreOp =
            static_cast<ThreadwiseWriteAllOp>(b.clone(*twop.getOperation()));
      } else {
        gemmStoreOp = twop;
      }
    }
  }

  if (gemmStoreOp) {
    if (ThreadwiseWriteAllOp twWriteAllOp =
            dyn_cast<ThreadwiseWriteAllOp>(gemmStoreOp)) {
      b.moveAfterIfNeeded(twWriteAllOp, copy);

      // 1. replace memref.copy with rock.threadwise_write_all
      target = cast<TypedValue<BaseMemRefType>>(
          applyViewsOnDest(b, loc, target, views));
      twWriteAllOp.getDestMutable().assign(target);

      b.eraseOp(copy);
      return success();
    }
  }

  return failure();
}

LogicalResult
ReduceRewritePattern::matchAndRewrite(rock::ReduceOp reduceOp,
                                      LinalgAlignRewriter &rewriter) const {
  Location loc = reduceOp.getLoc();
  SmallVector<TransformMapAttr, 4> views;
  auto threadwiseWriteOp = dyn_cast_if_present<ThreadwiseWriteAllOp>(
      traceToWriter(reduceOp.getIn(), views));
  if (!threadwiseWriteOp) {
    LLVM_DEBUG(llvm::dbgs() << "Not fusing reduction " << reduceOp
                            << " as it's not tied directly to a gemm\n");
    return success();
  }

  StoreMethodAttr stMethod;
  if (reduceOp.getReduceMethod() == ReduceMethod::Sum) {
    stMethod =
        StoreMethodAttr::get(rewriter.getContext(), StoreMethod::AtomicAdd);
  } else if (reduceOp.getReduceMethod() == ReduceMethod::Max) {
    stMethod =
        StoreMethodAttr::get(rewriter.getContext(), StoreMethod::AtomicMax);
  } else {
    // We are failing the pass here because rock.reduce appearing here means
    // we are committed to fusion and this is case, we cant handle (so far) in
    // this or a later pass.
    return reduceOp.emitError()
           << "Unsupported reduction type : " << reduceOp.getReduceMethodAttr();
  }

  if (threadwiseWriteOp.getStoreMethod() != rock::StoreMethod::Set) {
    // We are failing the pass here because another rock.reduce appearing here
    // means we are committed to fusion and this is case, we cant handle (so
    // far) in this or a later pass.
    return reduceOp.emitError("Another reduction op is not able to be fused "
                              "with a prior reduction op.");
  }

  bool isUniqueReader;
  LogicalResult checkResult = checkUniqueReader(
      reduceOp.getIn().getDefiningOp(), reduceOp, isUniqueReader);
  if (checkResult.failed()) {
    return checkResult;
  }
  if (!isUniqueReader) {
    threadwiseWriteOp = static_cast<ThreadwiseWriteAllOp>(
        rewriter.clone(*threadwiseWriteOp.getOperation()));
  }
  rewriter.moveAfterIfNeeded(threadwiseWriteOp, reduceOp);

  int64_t reductionAxis = reduceOp.getAxisAttr().getInt();
  TypedValue<ShapedType> redOut = reduceOp.getOut();
  ArrayRef<int64_t> reduceOutShape = redOut.getType().getShape();
  TypedValue<ShapedType> redIn = reduceOp.getIn();
  ArrayRef<int64_t> reduceInShape = redIn.getType().getShape();
  BottomUpTMBuilder dropReductionDim(rewriter, reduceOutShape, loc);
  for (uint32_t i = 0; i < reduceOutShape.size(); ++i) {
    if (i == reductionAxis) {
      dropReductionDim.broadcast({i}, {reduceInShape[i]});
    } else {
      dropReductionDim.passThrough({i}, {i});
    }
  }
  TransformMapAttr trAttr = dropReductionDim.get();
  views.push_back(trAttr);
  TypedValue<ShapedType> reduceOut = reduceOp.getOut();
  reduceOut = cast<TypedValue<ShapedType>>(
      applyViewsOnDest(rewriter, loc, reduceOut, views));
  threadwiseWriteOp.getDestMutable().assign(reduceOut);
  threadwiseWriteOp.setStoreMethodAttr(stMethod);

  rewriter.eraseOp(reduceOp);
  return success();
}

void RockLinalgAlignPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  func::FuncOp func = getOperation();
  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel"))
    return;
  {
    RewritePatternSet patterns(ctx);
    patterns.add<LAGenericRewritePattern, ReduceRewritePattern,
                 MemcpyRewritePattern>(ctx);
    if (failed(applyAlignPatterns(func, std::move(patterns))))
      return signalPassFailure();
  }
}
