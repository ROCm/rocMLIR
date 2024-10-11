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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"

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

  /// Move `toMove` before `latestPoint` unconditionally
  void moveBefore(Operation *toMove, Operation *latestOp);

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
  if (latestOp->isBeforeInBlock(toMove))
    toMove->moveBefore(latestOp);
  else
    LLVM_DEBUG(logger.startLine() << "   No move needed.\n");
}

void LinalgAlignRewriter::moveBefore(Operation *toMove, Operation *latestOp) {
  constexpr llvm::StringLiteral movePrefix("** Move    : ");
  constexpr llvm::StringLiteral beforePrefix("   before  : ");
  logOpActivity(movePrefix, toMove);
  logOpActivity(beforePrefix, latestOp);
  toMove->moveBefore(latestOp);
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

void LinalgAlignRewriter::notifyOperationInserted(Operation *op,
                                                  InsertPoint previous) {
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

void LinalgAlignRewriter::notifyBlockInserted(
    mlir::Block *block, mlir::Region *previous,
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
    if (isa<BlockArgument>(result)) {
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
                      Location loc, Type newElementType) {
  // 1. create a second allocation of the same type to hold loaded elements
  return b.create<GpuAllocOp>(
      loc, static_cast<MemRefType>(mrb.setElementType(newElementType)));
}

static void markGenericWritersToRevisit(LinalgAlignRewriter &b, Value rawSrc) {
  SmallVector<TransformMapAttr> views;
  auto genericWriter =
      dyn_cast_if_present<linalg::GenericOp>(traceToWriter(rawSrc, views));
  if (genericWriter)
    b.scheduleVisit(genericWriter);
}

template <typename TiledOp>
Value getRegisterValue(TiledOp op);
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
/// If `validityRecord` is a non-null pointer, create a value to record
/// whether each element of the tile was read from valid coordinates and
/// put that buffer into `validityRecord`.
///
/// Returns the new register tile.
template <typename TiledOp>
static Value
makeExtraInputTile(LinalgAlignRewriter &b, TiledOp tiledOp, Value src,
                   ArrayRef<TransformMapAttr> globalCoordsToGenericViews,
                   linalg::GenericOp laGeneric,
                   ValueRange dynamicValidities = {},
                   Value *validityRecord = nullptr) {
  // 0. capture the memref containing the outputs being written or
  // (in the case of propagating tiling informatinon up to gemm-independent
  // code) where the values will be written.
  Location loc = tiledOp.getLoc();
  Value tile = getRegisterValue(tiledOp);

  // move linalg.generic after the definitions of threadwiseReadIntoOp's
  // inputs to maintaine correct def-use chain.
  Operation *lastIdxDef = nullptr;
  for (Value idx : tiledOp.getExtraIndices()) {
    Operation *idxOp = idx.getDefiningOp();
    if (idxOp) {
      if (!lastIdxDef || (lastIdxDef->getBlock() == idxOp->getBlock() &&
                          lastIdxDef->isBeforeInBlock(idxOp))) {
        lastIdxDef = idxOp;
      }
    }
  }
  if (lastIdxDef && laGeneric->getBlock() == lastIdxDef->getBlock())
    b.moveAfterIfNeeded(laGeneric, lastIdxDef);

  // 1. create a second allocation of the same type to hold loaded elements
  // where the laGeneric is located.
  b.setInsertionPoint(laGeneric);
  auto mrbBuilder = cast<MemRefType>(tile.getType());
  MemRefType::Builder mrb(mrbBuilder);
  Value alloc = makeRegs(b, mrb, loc, getElementTypeOrSelf(src.getType()));

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

  // Reset the insertion point if laGeneric and tiledOp are in different blocks
  // and tiledOp reads from the output of linalg.generic. In such cases, we
  // should insert new transform operations and threadwiseReadIntoOp inside the
  // block of tiledOp, as their arguments may depend on that control flow. Later
  // we will also move linalg.generic into the block.
  if (laGeneric->getBlock() != tiledOp->getBlock() &&
      isa<ThreadwiseReadIntoOp>(tiledOp))
    b.setInsertionPoint(tiledOp);

  // 2.0. apply transform chain from output
  src = applyViewsOnDest(b, loc, src, globalCoordsToGenericViews);

  // 2.1. load into registers
  Type validityRecordResultType = vectorOfBoolShapedLike(alloc);
  ThreadwiseReadIntoOp threadwiseReadIntoOp = b.create<ThreadwiseReadIntoOp>(
      loc,
      validityRecord != nullptr ? TypeRange{validityRecordResultType}
                                : TypeRange{},
      src, alloc, dynamicValidities, tiledOp.getExtraViews(),
      /*extraIndices=*/tiledOp.getExtraIndices(), forceUnroll, useIndexDiffs);
  if (validityRecord != nullptr)
    *validityRecord = threadwiseReadIntoOp.getValidityRecord();

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

/// As above, but applying the tiling from a `threadwise_read_into`.
/// `validityRecords` is only populated if the tiling source accesses a
/// `validityRecords` parameter.
static void
addRegisterReadsForTiledOutput(LinalgAlignRewriter &b,
                               linalg::GenericOp laGeneric,
                               ThreadwiseReadIntoOp twReadOp,
                               ArrayRef<TransformMapAttr> relativeViewsOnResult,
                               SmallVectorImpl<Value> &newInputs,
                               SmallVectorImpl<Value> &validityRecords) {
  bool hasValidityRecord = twReadOp.getValidityRecord() != Value{};
  for (auto inp : laGeneric.getInputs()) {
    Value validityRecord = nullptr;
    Value newInput =
        makeExtraInputTile(b, twReadOp, inp, relativeViewsOnResult, laGeneric,
                           twReadOp.getDynamicValidities(),
                           hasValidityRecord ? &validityRecord : nullptr);
    newInputs.push_back(newInput);
    if (hasValidityRecord)
      validityRecords.push_back(validityRecord);
  }

  // move linalg.generic into the same block with threadwiseReadIntoOp
  if (laGeneric->getBlock() != twReadOp->getBlock())
    b.moveBefore(laGeneric, twReadOp);
}

static void reconfigureLAGeneric(LinalgAlignRewriter &b,
                                 linalg::GenericOp laGeneric,
                                 ValueRange newInputs, Value newOutput) {
  SmallVector<AffineMap, 5> lgAMaps;

  for (Value newInput : newInputs) {
    auto inpRank = cast<ShapedType>(newInput.getType()).getRank();
    lgAMaps.push_back(b.getMultiDimIdentityMap(inpRank));
  }

  laGeneric.getInputsMutable().assign(newInputs);
  laGeneric.getOutputsMutable().assign(newOutput);

  auto regRank = cast<ShapedType>(newOutput.getType()).getRank();

  lgAMaps.push_back(b.getMultiDimIdentityMap(regRank));
  laGeneric.setIndexingMapsAttr(b.getAffineMapArrayAttr(lgAMaps));

  // 2.3. Reset iterator types
  MLIRContext *ctx = b.getContext();
  SmallVector<Attribute, 5> iteratorTypes;
  iteratorTypes.resize(regRank, linalg::IteratorTypeAttr::get(
                                    ctx, utils::IteratorType::parallel));
  laGeneric.setIteratorTypesAttr(ArrayAttr::get(ctx, iteratorTypes));
}

static LogicalResult canFuseAcrossAtomic(LinalgAlignRewriter &b,
                                         linalg::GenericOp laGeneric) {
  auto opCanSwapWithAtomic = [](Operation &op) -> bool {
    return llvm::TypeSwitch<Operation &, bool>(op)
        .Case<linalg::YieldOp>([](linalg::YieldOp ignored) { return true; })
        .Case<arith::TruncFOp>([](arith::TruncFOp truncOp) {
          Type resultType = truncOp.getOut().getType();
          return isa<Float32Type, Float16Type>(resultType);
        })
        .Case<arith::TruncIOp>([](arith::TruncIOp truncOp) {
          return truncOp.getOut().getType().isInteger(32);
        })
        .Default([](Operation &ignored) { return false; });
  };
  return success(
      llvm::all_of(laGeneric.getRegion().getOps(), opCanSwapWithAtomic));
}

/// Return true if all the operations inside a given `linalg.generic` are known
/// to preserve 0 - that is, they return zero if all their non-constant inputs
/// are zero. This property allows us to not need to re-apply any padding that's
/// being moved from the outputs of the generic to the inputs because we know
/// that if the inputs all fall into the padding, the result of the elementwise
/// function will also be the expected zero.
static LogicalResult knownToPreserveZero(linalg::GenericOp laGeneric,
                                         LinalgAlignRewriter &b) {
  // Brute-force test: clone the generic, replace all the arguments with 0s,
  // and constant-fold.
  LLVM_DEBUG(llvm::dbgs() << "* Cloning generic to test if it allows 0s\n");
  auto clonedOp = cast<linalg::GenericOp>(b.clone(*laGeneric));
  Location loc = clonedOp.getLoc();
  LinalgAlignRewriter::InsertionGuard guard(b);
  b.setInsertionPointToStart(&clonedOp.getRegion().front());
  OperationFolder folder(clonedOp.getContext(), b.getListener());
  for (BlockArgument &arg : clonedOp.getRegion().getArguments()) {
    Value zero = createZeroConstantOp(b, loc, arg.getType());
    arg.replaceAllUsesWith(zero);
  }
  for (Operation &op :
       llvm::make_early_inc_range(clonedOp.getRegion().getOps())) {
    bool ignored = false;
    (void)folder.tryToFold(&op, &ignored);
  }
  LLVM_DEBUG(llvm::dbgs() << "* Folded input fusion region to " << clonedOp
                          << "\n");
  auto yieldOp =
      cast<linalg::YieldOp>(clonedOp.getRegion().front().getTerminator());
  bool foldedToZero = llvm::all_of(yieldOp.getValues(), [&](Value v) {
    return matchPattern(v, m_AnyZeroFloat()) || matchPattern(v, m_Zero());
  });
  LLVM_DEBUG(llvm::dbgs() << "* Cloning generic to test if it allows 0s\n");
  b.eraseOp(clonedOp);
  return success(foldedToZero);
}

/// If this generic doesn't preserve zero (ex, it's x => x + 1) and if the
/// validity of the tiling operation was being tracked (this indicates input
/// fusion), then:
/// - Clone the output tile
/// - Set up a register->register threadwise_read_into between this cloned tile
///   and the original output tile, with dynamic validities drawn from the
///   validity results of each read.
/// The extra threadwise_read_into we create here will cause elements that
/// didn't actually get fetched from memory to become 0s again thanks to an if
/// statement in what would otherwise be a memcpy().
///
/// Returns the validity record from the padding read if there is one.
static std::optional<Value>
reapplyPaddingIfNeeded(linalg::GenericOp reconfiguredGeneric,
                       ValueRange validityRecords,
                       ThreadwiseReadIntoOp oldTwRead, LinalgAlignRewriter &b) {
  // If the old read never produces validity records, we just need to erase it.
  if (!oldTwRead.getValidityRecord())
    return std::nullopt;
  // However, if we don't need to reapply the mask, we can return the null
  // result to "replace" all zero uses of the validity record. Note that if the
  // validity record is used, we'll still need to construct the read.
  if (oldTwRead.getValidityRecord().use_empty()) {
    if (validityRecords.empty())
      return Value{};
    if (succeeded(knownToPreserveZero(reconfiguredGeneric, b)))
      return Value{};
  }
  assert(reconfiguredGeneric.getOutputs().size() == 1 &&
         "Multi-output generics shouldn't have made it here since they're not "
         "supported");
  Value originalTile = reconfiguredGeneric.getOutputs()[0];
  LinalgAlignRewriter::InsertionGuard guard(b);
  b.setInsertionPoint(reconfiguredGeneric);
  Value unmaskedTile = b.clone(*originalTile.getDefiningOp())->getResult(0);
  b.modifyOpInPlace(reconfiguredGeneric, [&]() {
    reconfiguredGeneric.getOutputsMutable()[0].assign(unmaskedTile);
  });
  b.setInsertionPointAfter(reconfiguredGeneric);
  auto maskingRead = b.create<rock::ThreadwiseReadIntoOp>(
      reconfiguredGeneric.getLoc(), vectorOfBoolShapedLike(unmaskedTile),
      unmaskedTile, originalTile,
      /*dynamicValidities=*/validityRecords,
      /*extraViews=*/b.getArrayAttr({}), /*extraIndices=*/ValueRange{},
      /*forceUnroll=*/false, /*useIndexDiffs=*/false);
  return maskingRead.getValidityRecord();
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

  if (gemmStoreOp) {
    if (gemmStoreOp.getStoreMethod() != rock::StoreMethod::Set) {
      if (failed(canFuseAcrossAtomic(b, laGeneric))) {
        return laGeneric.emitOpError(
            "is infusible with non-`Set` store method");
      }
    }
  }

  // 1.2. If there is no input being written, try to find a threadwise_read_into
  // operation that reads from the output of this generic. If there is such an
  ThreadwiseReadIntoOp tileReadOp;
  if (!gemmStoreOp) {
    if (!out.getDefiningOp<memref::AllocOp>()) {
      return b.notifyMatchFailure(
          loc, "generic output is not suitable for input fusion\n");
    }
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
    // Only populated if this threadwise_read_into is tracking its validity -
    // that is, if this is part of an input fusion (where we're worried about
    // re-applying padding in cases where the generic doesn't preserve 0).
    SmallVector<Value> newValidityRecords;
    addRegisterReadsForTiledOutput(b, laGeneric, tileReadOp,
                                   globalCoordsToGenericViews, newInputs,
                                   newValidityRecords);
    // Prevent SSA weirdness from register allocations introduced too late.
    // addRegisterReadsForTiledOutput() may have moved laGeneric into a
    // different block. In this case, SSA is already in good shape.
    if (newOutput.getDefiningOp()->getBlock() == laGeneric->getBlock())
      b.moveBeforeIfNeeded(newOutput.getDefiningOp(), laGeneric);
    reconfigureLAGeneric(b, laGeneric, newInputs, newOutput);
    std::optional<Value> newValidityRecord =
        reapplyPaddingIfNeeded(laGeneric, newValidityRecords, tileReadOp, b);
    if (!newValidityRecord)
      b.eraseOp(tileReadOp);
    else
      b.replaceOp(tileReadOp, *newValidityRecord);
    return success();
  }
  auto outType = cast<ShapedType>(out.getType());
  auto inpType = cast<ShapedType>(laGenericArgLeadingToTile.getType());
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
  auto gemmOutType = cast<MemRefType>(gemmOutRegs.getType());

  // 3.1. Make an allocation that matches the tile but has the type of the
  // linalg.generic output.
  MemRefType::Builder mrb(gemmOutType);
  Value laOutRegs = makeRegs(b, mrb, loc, getElementTypeOrSelf(out.getType()));

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

// This function will attempt to add blockwise reductions when fusing
// in reduction to the write back of the core kernel.
static LogicalResult insertBlockwiseReduction(
    LinalgAlignRewriter &rewriter, Location loc, rock::ReduceOp reduceOp,
    ThreadwiseWriteAllOp threadwiseWriteOp, StoreMethodAttr stMethod) {
  // This has < block dimensions ... > x tid x iter to Gemm Dimensions.
  ArrayAttr extraViews = threadwiseWriteOp.getExtraViews();
  ArrayAttr destTrs;
  Value dest;
  std::tie(dest, destTrs, std::ignore) =
      untransform(rewriter, threadwiseWriteOp.getDest());

  ArrayAttr toBeReducedViews = prependUpperViews(rewriter, extraViews, destTrs);
  TransformMapAttr firstCoordTransform =
      cast<TransformMapAttr>(toBeReducedViews[0]);
  int64_t upperRank = firstCoordTransform.getUpperBounds().size();
  SetVector<int64_t> removeIndicesSet;
  // We only want to keep tid x iter in the maps
  // which is the last two for block subtile
  for (int64_t i = 0; i < upperRank - 2; i++) {
    removeIndicesSet.insert(i);
  }
  FailureOr<ArrayAttr> blockSubTileViews =
      removeUpperDims(rewriter, toBeReducedViews, removeIndicesSet);
  if (failed(blockSubTileViews)) {
    LLVM_DEBUG(llvm::dbgs() << "blockSubTileViews creation using "
                               "removeUpperDims is unsuccesful.\n");
    return failure();
  }
  // We only want to keep tid in the maps
  // which is the last two for block subtile tid
  // hence, add back iter to remove indices.
  removeIndicesSet.insert(upperRank - 1);

  FailureOr<ArrayAttr> blockSubTileTidSliceViews =
      removeUpperDims(rewriter, toBeReducedViews, removeIndicesSet);
  if (failed(blockSubTileTidSliceViews)) {
    LLVM_DEBUG(llvm::dbgs() << "blockSubTileTidSliceViews creation using "
                               "removeUpperDims is unsuccesful.\n");
    return failure();
  }
  // We only want to keep iter in the maps
  // which is the last one.
  removeIndicesSet.remove(upperRank - 1);
  removeIndicesSet.insert(upperRank - 2);

  FailureOr<ArrayAttr> threadSubTileViews =
      removeUpperDims(rewriter, toBeReducedViews, removeIndicesSet);
  if (failed(threadSubTileViews)) {
    LLVM_DEBUG(llvm::dbgs() << "threadSubTileViews creation using "
                               "removeUpperDims is unsuccesful.\n");
    return failure();
  }

  // Extract grid-only dims
  removeIndicesSet.clear();
  for (int64_t i = upperRank - 2; i < upperRank; i++) {
    removeIndicesSet.insert(i);
  }
  FailureOr<ArrayAttr> gridOnlyDims =
      removeUpperDims(rewriter, toBeReducedViews, removeIndicesSet);
  if (failed(gridOnlyDims)) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "gridOnlyDims creation using removeUpperDims is unsuccesful.\n");
    return failure();
  }

  SmallVector<int64_t> gridOnlyDimIdxs;
  for (int64_t i = 0; i < upperRank - 2; i++) {
    gridOnlyDimIdxs.push_back(i);
  }
  FailureOr<llvm::SmallDenseMap<int64_t, SmallVector<SubDimInfo>>>
      lowerSubDims =
          getLowerSubDimensions(rewriter, toBeReducedViews, gridOnlyDimIdxs);
  if (failed(lowerSubDims)) {
    LLVM_DEBUG(llvm::dbgs() << "lowerSubDims creation using "
                               "getLowerSubDimensions is unsuccesful.\n");
    return failure();
  }

  int64_t reductionAxis = reduceOp.getAxisAttr().getInt();
  TypedValue<ShapedType> redOut = reduceOp.getOut();
  ArrayRef<int64_t> reduceOutShape = redOut.getType().getShape();
  TypedValue<ShapedType> redIn = reduceOp.getIn();
  ArrayRef<int64_t> reduceInShape = redIn.getType().getShape();

  int64_t blockReductionAxis = reductionAxis;
  int64_t blockReductionAxisFromLeft =
      (reduceInShape.size() - 1) - blockReductionAxis;

  ArrayRef<int64_t> blockLowerShape = getLowerShape(blockSubTileViews.value());
  ArrayRef<int64_t> blockSubTileTidSliceShape =
      getLowerShape(blockSubTileTidSliceViews.value());
  int64_t blockSubTileTidSliceRank = blockSubTileTidSliceShape.size();
  // The block sub-tile view might not have the slower changing
  // dimensions in it. Thus, we always keep track of the reduction
  // dimensions from its distance to fastest changing dimensions.
  blockReductionAxis =
      blockSubTileTidSliceRank - 1 - blockReductionAxisFromLeft;
  int64_t partialReductionsPerThread =
      blockSubTileTidSliceShape[blockReductionAxis];
  int64_t ldsWorkspaceSize = 1;
  for (auto [idx, size] : llvm::enumerate(blockLowerShape)) {
    if (idx == (size_t)blockReductionAxis) {
      ldsWorkspaceSize *= partialReductionsPerThread;
    } else {
      ldsWorkspaceSize *= size;
    }
  }
  auto maybeArch = getArch(reduceOp);
  if (succeeded(maybeArch)) {
    if (failed(checkLDSSize(maybeArch.value(), ldsWorkspaceSize))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "lds size for blockwise reduction does not fit.\n");
      return failure();
    }
  }
  TypedValue<MemRefType> src = threadwiseWriteOp.getSource();
  auto broadcastReducedSrc = rewriter.create<GpuAllocOp>(loc, src.getType());
  Value ldsWorkspace = rock::gpuAlloc(rewriter, loc, ldsWorkspaceSize,
                                      src.getType().getElementType(),
                                      gpu::AddressSpace::Workgroup);

  rewriter.create<BlockwiseBroadcastReduceOp>(
      loc, src, ldsWorkspace, broadcastReducedSrc,
      /*extraOut=*/nullptr, rewriter.getIndexAttr(blockReductionAxis),
      reduceOp.getReduceMethodAttr(), blockSubTileViews.value(),
      blockSubTileTidSliceViews.value(), threadSubTileViews.value(),
      /*extraViews=*/nullptr,
      getBlockSize(reduceOp->getParentOfType<func::FuncOp>()).value());

  ViewLikeOpInterface viewOp =
      ldsWorkspace.getDefiningOp<ViewLikeOpInterface>();
  rewriter.create<GpuDeallocOp>(loc, viewOp.getViewSource());
  // Create partial reduction views
  ArrayAttr paddedReducedTrStack;
  {
    SmallVector<Attribute> transformAttrs;
    ArrayRef<int64_t> blockTileShape = getLowerShape(blockSubTileViews.value());
    SmallVector<SmallString<8>> names =
        createDimNames(blockTileShape.size(), "dim");
    SmallVector<StringRef> nameRefs = getStringRefsFor(names);
    TopDownTMBuilder toReducedView(rewriter, nameRefs, blockTileShape);
    for (unsigned i = 0; i < blockTileShape.size(); i++) {
      if (blockReductionAxis == i) {
        // The blockwise_broadcast_reduce will populate
        // all indices of pre-reduction space with the
        // reduced value. However, for the write back
        // we only want one of the reduced values to be
        // written. Therefore, we keep the 0th and declare
        // rest as padding.
        toReducedView.pad({nameRefs[i]}, {0, blockTileShape[i] - 1});
      } else {
        toReducedView.passThrough({nameRefs[i]}, {i}, {nameRefs[i]});
      }
    }
    transformAttrs.push_back(toReducedView.get());
    ArrayAttr arrayTransformAttrs = rewriter.getArrayAttr(transformAttrs);
    paddedReducedTrStack = prependUpperViews(
        rewriter, blockSubTileViews.value(), arrayTransformAttrs);
  }

  // Recombine block dimensions
  {
    SmallVector<TransformMapAttr> transformAttrs;
    ArrayRef<int64_t> lowerShapeGridOnly = getLowerShape(gridOnlyDims.value());
    size_t lowerGridOnlyRank = lowerShapeGridOnly.size();
    for (auto [idx, attr] : llvm::enumerate(paddedReducedTrStack)) {
      TransformMapAttr trMapAttr = cast<TransformMapAttr>(attr);
      SmallVector<TransformAttr> trAttrs;
      ArrayRef<int64_t> gridUpperShape;
      ArrayRef<int64_t> gridLowerShape;
      if (idx < gridOnlyDims.value().size()) {
        TransformMapAttr gridOnlyAttr =
            cast<TransformMapAttr>(gridOnlyDims.value()[idx]);
        ArrayRef<TransformAttr> ops = gridOnlyAttr.getOps();
        trAttrs.insert(trAttrs.end(), ops.begin(), ops.end());
        gridUpperShape = gridOnlyAttr.getUpperBounds().asArrayRef();
        gridLowerShape = gridOnlyAttr.getLowerBounds().asArrayRef();
      } else {
        SmallVector<SmallString<8>> names =
            createDimNames(lowerGridOnlyRank, "dim");
        SmallVector<StringRef> nameRefs = getStringRefsFor(names);
        SmallVector<unsigned> dims;
        for (unsigned i = 0; i < lowerGridOnlyRank; i++) {
          dims.push_back(i);
        }
        gridUpperShape = lowerShapeGridOnly;
        gridLowerShape = lowerShapeGridOnly;
        TransformAttr blockPt = TransformAttr::get(
            rewriter.getContext(), TransformType::PassThrough, {}, nameRefs,
            dims, nameRefs, dims);
        trAttrs.push_back(blockPt);
      }
      for (TransformAttr trAttr : trMapAttr.getOps()) {
        SmallVector<unsigned> upperDims;
        llvm::transform(
            trAttr.getUpperDims(), std::back_inserter(upperDims),
            [&](unsigned idx) { return idx + gridUpperShape.size(); });
        SmallVector<unsigned> lowerDims;
        llvm::transform(
            trAttr.getLowerDims(), std::back_inserter(lowerDims),
            [&](unsigned idx) { return idx + gridLowerShape.size(); });
        TransformAttr newTrAttr =
            TransformAttr::get(rewriter.getContext(), trAttr.getType(),
                               trAttr.getParams(), trAttr.getUpperNames(),
                               upperDims, trAttr.getLowerNames(), lowerDims);
        trAttrs.push_back(newTrAttr);
      }
      // set the bounds
      SmallVector<int64_t> upperBounds = llvm::to_vector(gridUpperShape);
      ArrayRef<int64_t> origUpperBounds =
          trMapAttr.getUpperBounds().asArrayRef();
      upperBounds.insert(upperBounds.end(), origUpperBounds.begin(),
                         origUpperBounds.end());
      SmallVector<int64_t> lowerBounds = llvm::to_vector(gridLowerShape);
      ArrayRef<int64_t> origLowerBounds =
          trMapAttr.getLowerBounds().asArrayRef();
      lowerBounds.insert(lowerBounds.end(), origLowerBounds.begin(),
                         origLowerBounds.end());
      // create new trMapAttr
      LLVM_DEBUG(llvm::dbgs() << "trAttrs = ";
                 llvm::interleaveComma(trAttrs, llvm::dbgs());
                 llvm::dbgs() << "\n"; llvm::dbgs() << "upperBounds = ";
                 llvm::interleaveComma(upperBounds, llvm::dbgs());
                 llvm::dbgs() << "\n"; llvm::dbgs() << "lowerBounds = ";
                 llvm::interleaveComma(lowerBounds, llvm::dbgs());
                 llvm::dbgs() << "\n");
      TransformMapAttr newTrMap =
          TransformMapAttr::get(trAttrs, upperBounds, lowerBounds);
      transformAttrs.push_back(newTrMap);
    }
    ArrayRef<int64_t> currLowerShape =
        cast<TransformMapAttr>(transformAttrs.back()).getLowerBounds();
    if (currLowerShape.size() < lowerGridOnlyRank * 2) {
      SmallVector<SmallString<8>> names =
          createDimNames(currLowerShape.size(), "d");
      SmallVector<StringRef> nameRefs = getStringRefsFor(names);
      TopDownTMBuilder toAddMissingBlockDims(rewriter, nameRefs,
                                             currLowerShape);
      {
        SmallVector<unsigned> gridOnlyDimIdxs;
        for (unsigned i = 0; i < upperRank - 2; i++) {
          gridOnlyDimIdxs.push_back(i);
        }
        toAddMissingBlockDims.passThrough(gridOnlyDimIdxs, gridOnlyDimIdxs);
        int64_t missingDimCount = lowerGridOnlyRank * 2 - currLowerShape.size();
        SmallVector<SmallString<8>> names =
            createDimNames(missingDimCount, "cd");
        SmallVector<StringRef> nameRefs = getStringRefsFor(names);
        unsigned dimInsertionPoint = 3;
        for (int64_t md = 0; md < missingDimCount; md++) {
          toAddMissingBlockDims.constDim(nameRefs.back(), dimInsertionPoint++,
                                         0, 1);
        }
        for (unsigned lowerDim = 3; lowerDim < currLowerShape.size();
             lowerDim++) {
          toAddMissingBlockDims.passThrough({dimInsertionPoint++}, {lowerDim});
        }
        TransformMapAttr addMissingBlockDims = toAddMissingBlockDims.get();
        LLVM_DEBUG(llvm::dbgs()
                   << "addMissingBlockDims = " << addMissingBlockDims << "\n");
        transformAttrs.push_back(addMissingBlockDims);
      }
    }
    currLowerShape =
        cast<TransformMapAttr>(transformAttrs.back()).getLowerBounds();
    if (currLowerShape.size() != lowerGridOnlyRank * 2) {
      LLVM_DEBUG(llvm::dbgs() << "Recombine: currLowerRank="
                              << currLowerShape.size() << "\n");
      LLVM_DEBUG(llvm::dbgs() << "Recombine: lowerGridOnlyRank="
                              << lowerGridOnlyRank << "\n");
      LLVM_DEBUG(llvm::dbgs() << "Recombine: current lower rank should be 2x "
                                 "as the grid only rank\n");
      return failure();
    }

    // The last two transforms are constructed bottom up as it is easier.
    // where we joint them once we have grid and block tiles coordinates
    // seperated.
    ArrayRef<int64_t> toBeReducedShape = getLowerShape(toBeReducedViews);
    SmallVector<SmallString<8>> reduceLowerShapeNames =
        createDimNames(toBeReducedShape.size(), "d");
    SmallVector<StringRef> reduceLowerShapeNameRefs =
        getStringRefsFor(reduceLowerShapeNames);
    BottomUpTMBuilder toMatrixView(rewriter, reduceLowerShapeNameRefs,
                                   toBeReducedShape);
    llvm::SmallDenseMap<int64_t, SmallVector<int64_t>> gridSubDims;
    llvm::SmallDenseMap<int64_t, SmallVector<int64_t>> blockSubDims;
    TransformMapAttr lastMerge;
    {
      llvm::SmallDenseMap<int64_t, SmallVector<SmallString<8>>> names;
      llvm::SmallDenseMap<int64_t, SmallVector<StringRef>> nameRefs;
      llvm::SmallDenseMap<int64_t, SmallVector<unsigned>> upperDims;
      int64_t dimInsertionPoint = 0;
      for (unsigned dim = 0; dim < toBeReducedShape.size(); dim++) {
        // The lower subDims contain sub-dimensions where blocking
        // indices -- namely g_block, m_block and n_block -- maps to
        // in the matrix coordinates. Here we split out matrix dims
        // into sub-dims that are related to the said blocking dimensions.
        SmallVector<SubDimInfo> subDims = lowerSubDims.value()[dim];
        llvm::sort(subDims, [](const SubDimInfo &L, const SubDimInfo &R) {
          return L.stride > R.stride;
        });
        SmallVector<int64_t> splitSizes;
        int64_t currSize = toBeReducedShape[dim];
        for (const SubDimInfo &subDim : subDims) {
          if (currSize % (subDim.size * subDim.stride) != 0) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Recombine: currSize=" << currSize << "\n");
            LLVM_DEBUG(llvm::dbgs()
                       << "Recombine: subDim.size=" << subDim.size << "\n");
            LLVM_DEBUG(llvm::dbgs()
                       << "Recombine: subDim.stride=" << subDim.stride << "\n");
            LLVM_DEBUG(
                llvm::dbgs()
                << "Recombine: subDims should equally divide current dims\n");
            return failure();
          }
          int64_t newSize = currSize / (subDim.size * subDim.stride);
          if (newSize > 1) {
            blockSubDims[dim].push_back(dimInsertionPoint);
            SmallString<8> dimName(
                Twine("block_dim" + Twine(dim) + "_" + Twine(dimInsertionPoint))
                    .str());
            names[dim].push_back(dimName);
            nameRefs[dim].push_back(names[dim].back());
            upperDims[dim].push_back(dimInsertionPoint++);
            splitSizes.push_back(newSize);
          }
          gridSubDims[dim].push_back(dimInsertionPoint);
          SmallString<8> dimName(
              Twine("grid_dim" + Twine(dim) + "_" + Twine(dimInsertionPoint))
                  .str());
          names[dim].push_back(dimName);
          nameRefs[dim].push_back(names[dim].back());
          upperDims[dim].push_back(dimInsertionPoint++);
          splitSizes.push_back(subDim.size);
          currSize = subDim.stride;
        }
        if (currSize > 1 || splitSizes.empty()) {
          blockSubDims[dim].push_back(dimInsertionPoint);
          SmallString<8> dimName(
              Twine("block_dim" + Twine(dim) + "_" + Twine(dimInsertionPoint))
                  .str());
          names[dim].push_back(dimName);
          nameRefs[dim].push_back(names[dim].back());
          upperDims[dim].push_back(dimInsertionPoint++);
          splitSizes.push_back(currSize);
        }
        LLVM_DEBUG(llvm::dbgs() << "dim=" << dim << "\n");
        LLVM_DEBUG(llvm::dbgs() << "\tsplits=";
                   llvm::interleaveComma(splitSizes, llvm::dbgs());
                   llvm::dbgs() << "\n");
        toMatrixView.unmerge(nameRefs[dim], upperDims[dim],
                             reduceLowerShapeNameRefs[dim], splitSizes);
      }
      lastMerge = toMatrixView.get();
    }
    LLVM_DEBUG(llvm::dbgs() << "lastMerge=" << lastMerge << "\n");
    // The above view contains splitted sub-dims that are either associated
    // with grid and non-grid dimensions. Then, we concat them as follows:
    // [concat_grid_dim0, concat_grid_dim1, .. , concat_grid_dimX,
    // concat_blk_dim0, concat_blk_dim1, .. , concat_blk_dimX]
    BottomUpTMBuilder toGridBlockSeperation =
        BottomUpTMBuilder::above(toMatrixView, lastMerge);
    TransformMapAttr gridblockSeperation;
    {
      SmallVector<StringRef, 4> lowerNameRefs;
      toGridBlockSeperation.getStartNames(lowerNameRefs);
      SmallVector<std::string> upperGridNames;
      for (unsigned dim = 0; dim < toBeReducedShape.size(); dim++) {
        upperGridNames.push_back(Twine("grid_dim" + Twine(dim)).str());
        if (gridSubDims.contains(dim)) {
          SmallVector<StringRef, 4> upperGridSubDimNames;
          for (int64_t upperGridSubDim : gridSubDims[dim]) {
            upperGridSubDimNames.push_back(lowerNameRefs[upperGridSubDim]);
          }
          toGridBlockSeperation.merge(upperGridNames.back(), dim,
                                      upperGridSubDimNames);
        } else {
          toGridBlockSeperation.addDim(upperGridNames.back(), dim, 1);
        }
      }
      SmallVector<std::string> upperBlockNames;
      for (unsigned dim = 0; dim < toBeReducedShape.size(); dim++) {
        upperBlockNames.push_back(Twine("block_dim" + Twine(dim)).str());
        if (blockSubDims.contains(dim)) {
          SmallVector<StringRef, 4> upperBlockSubDimNames;
          for (int64_t upperBlockSubDim : blockSubDims[dim]) {
            upperBlockSubDimNames.push_back(lowerNameRefs[upperBlockSubDim]);
          }
          toGridBlockSeperation.merge(upperBlockNames.back(),
                                      dim + toBeReducedShape.size(),
                                      upperBlockSubDimNames);
        } else {
          toGridBlockSeperation.addDim(upperBlockNames.back(),
                                       dim + toBeReducedShape.size(), 1);
        }
      }
      gridblockSeperation = toGridBlockSeperation.get();
    }
    LLVM_DEBUG(llvm::dbgs()
               << "gridblockSeperation=" << gridblockSeperation << "\n");
    // Now we join them to finish the recombination.
    transformAttrs.push_back(gridblockSeperation);
    transformAttrs.push_back(lastMerge);
    reduceInShape =
        cast<TransformMapAttr>(transformAttrs.back()).getLowerBounds();
    BottomUpTMBuilder dropReductionDim(rewriter, reduceOutShape, loc);
    for (uint32_t i = 0; i < reduceOutShape.size(); ++i) {
      if (i == reductionAxis) {
        dropReductionDim.broadcast({i}, {reduceInShape[i]});
      } else {
        dropReductionDim.passThrough({i}, {i});
      }
    }
    transformAttrs.push_back(dropReductionDim.get());
    threadwiseWriteOp.setExtraViewsAttr(rewriter.getArrayAttr({}));
    threadwiseWriteOp.getSourceMutable().assign(broadcastReducedSrc);
    LLVM_DEBUG(llvm::dbgs() << "transformAttrs = "
                            << "\n";
               llvm::interleaveComma(transformAttrs, llvm::dbgs());
               llvm::dbgs() << "\n");
    TypedValue<ShapedType> reduceOut = reduceOp.getOut();
    reduceOut = cast<TypedValue<ShapedType>>(
        applyViewsOnDest(rewriter, loc, reduceOut, transformAttrs));
    threadwiseWriteOp.getDestMutable().assign(reduceOut);
    // TODO : in future if all reductions are done within the block
    // we can revert this back to a non-atomic store.
    threadwiseWriteOp.setStoreMethodAttr(stMethod);
  }
  return success();
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

  LogicalResult canUseBlockwiseReductions = insertBlockwiseReduction(
      rewriter, loc, reduceOp, threadwiseWriteOp, stMethod);
  // fallback to doing pure atomics based reductions
  if (failed(canUseBlockwiseReductions)) {
    LLVM_DEBUG(llvm::dbgs() << "Unable to add blockwise reductions for this "
                               "reduction fusion case.\n");
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
    LLVM_DEBUG(llvm::dbgs() << "views = "
                            << "\n";
               llvm::interleaveComma(views, llvm::dbgs());
               llvm::dbgs() << "\n");
    TypedValue<ShapedType> reduceOut = reduceOp.getOut();
    reduceOut = cast<TypedValue<ShapedType>>(
        applyViewsOnDest(rewriter, loc, reduceOut, views));
    threadwiseWriteOp.getDestMutable().assign(reduceOut);
    threadwiseWriteOp.setStoreMethodAttr(stMethod);
  }
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
