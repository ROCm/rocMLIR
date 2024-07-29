//===- Regularize.cpp - rewrites to allow Rock kernel fusion  ------===//
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
#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREGULARIZEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-regularize"

using namespace mlir;
using namespace mlir::rock;

namespace {

////////////////////////////////////////////////////////////////////////
////  Convert memref.collapse/expand_shape ops to rock.transform
////////////////////////////////////////////////////////////////////////
struct CollapseRewritePattern
    : public OpRewritePattern<memref::CollapseShapeOp> {
  using OpRewritePattern<memref::CollapseShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CollapseShapeOp collapseOp,
                                PatternRewriter &rw) const final {
    Location loc = collapseOp.getLoc();
    ArrayRef<int64_t> inpShape = collapseOp.getSrcType().getShape();
    ArrayRef<int64_t> outShape = collapseOp.getResultType().getShape();
    SmallVector<ReassociationIndices, 4> reassocs =
        collapseOp.getReassociationIndices();

    rock::TransformMapAttr transform =
        rock::transformCollapseShape(rw, loc, inpShape, outShape, reassocs);
    if (!transform)
      return rw.notifyMatchFailure(
          loc, "could not translate memref collapse into rock transform");
    rw.replaceOpWithNewOp<rock::TransformOp>(collapseOp, collapseOp.getSrc(),
                                             transform);
    return success();
  }
};

struct ExpandRewritePattern : public OpRewritePattern<memref::ExpandShapeOp> {
  using OpRewritePattern<memref::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExpandShapeOp expandOp,
                                PatternRewriter &rw) const final {
    Location loc = expandOp.getLoc();
    ArrayRef<int64_t> inpShape = expandOp.getSrcType().getShape();
    ArrayRef<int64_t> outShape = expandOp.getResultType().getShape();
    SmallVector<ReassociationIndices, 4> reassocs =
        expandOp.getReassociationIndices();

    rock::TransformMapAttr transform =
        rock::transformExpandShape(rw, loc, inpShape, outShape, reassocs);
    if (!transform)
      return rw.notifyMatchFailure(
          loc, "could not translate memref expansion into rock transform");
    rw.replaceOpWithNewOp<rock::TransformOp>(expandOp, expandOp.getSrc(),
                                             transform);
    return success();
  }
};

////////////////////////////////////////////////////////////////////////
////  Test linalg.generic for regularity
////////////////////////////////////////////////////////////////////////
static bool isRegularGeneric(linalg::GenericOp lgop) {
  // parallel
  for (utils::IteratorType iterType : lgop.getIteratorTypesArray()) {
    if (!linalg::isParallelIterator(iterType))
      return false; //"Only fully parallel supported"
  }

  // 1 output
  auto outs = lgop.getOutputs();
  if (outs.size() > 1)
    return false; //"Only 1 output supported"

  // all index maps must be identity
  auto idxMaps = lgop.getIndexingMapsArray();
  auto outIdxMap = idxMaps.back();
  if (!outIdxMap.isIdentity()) {
    return false; //"Only output identity map supported"
  }

  for (auto idxMap : idxMaps) {
    if (idxMap != outIdxMap)
      return false; //"Must be same index maps"
  }
  return true;
}

////////////////////////////////////////////////////////////////////////
////  Regularize linalg.generic inputs
////////////////////////////////////////////////////////////////////////
struct RegularizeGenericRewritePattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp lgop,
                                PatternRewriter &rw) const override {
    LogicalResult lres = failure();

    // parallel
    for (utils::IteratorType iterType : lgop.getIteratorTypesArray()) {
      if (!linalg::isParallelIterator(iterType))
        return lgop.emitError("Only fully parallel supported");
    }

    // apply transforms to inputs
    lres = makeLinalgGenericWithIdentityAffMaps(rw, lgop);
    return lres;
  }
};

static void annotateGenericOp(linalg::GenericOp lgop) {
  MLIRContext *ctx = lgop.getContext();
  int64_t majorTensorSize = 0;
  size_t majorTensorIdx;
  size_t argIdx = -1;
  if (lgop.getInputs().size() == 1) {
    lgop->setAttr("rock.majorTensorNumber",
                  IntegerAttr::get(IndexType::get(ctx), 1));
    return;
  }
  for (auto [inputIdx, inp] : llvm::enumerate(lgop.getInputs())) {
    while (auto viewOp =
               dyn_cast_or_null<ViewLikeOpInterface>(inp.getDefiningOp()))
      inp = viewOp.getViewSource();

    if (isa<BlockArgument>(inp)) {
      auto arg = dyn_cast<BlockArgument>(inp);
      auto shape = cast<ShapedType>(inp.getType());
      int64_t argSize = shape.getNumElements();
      if (inputIdx == 0 || argSize > majorTensorSize ||
          (argSize == majorTensorSize && argIdx > arg.getArgNumber())) {
        majorTensorIdx = inputIdx;
        majorTensorSize = argSize;
        argIdx = arg.getArgNumber();
      }
    }
  }
  if (majorTensorIdx >= 0)
    lgop->setAttr("rock.majorTensorNumber",
                  IntegerAttr::get(IndexType::get(ctx), majorTensorIdx));
}

////////////////////////////////////////////////////////////////////////
////  Classify buffer writes into input and output fusion chains
////////////////////////////////////////////////////////////////////////

namespace {
struct TransformPushState {
  llvm::SmallVector<memref::AllocOp> worklist;
  llvm::SmallPtrSet<OpOperand *, 8> inputFusionWrites;
  llvm::SmallPtrSet<OpOperand *, 8> outputFusionWrites;
};
} // end namespace

/// Given an operand that reads a buffer, collect the writer of thet buffer and,
/// recursively, all the writers of the buffers that writer reads, and so on.
/// This is meant to collect all the buffer writes involved in an input fusion.
static void collectInputFusionWrites(OpOperand *reader,
                                     const BufferDependencyAnalysis &bufferDeps,
                                     TransformPushState &state) {
  std::optional<memref::AllocOp> maybeBuffer = bufferDeps.getReadBuffer(reader);
  // This reads an argument or some such thing
  if (!maybeBuffer)
    return;
  std::optional<SmallVector<OpOperand *>> writers =
      bufferDeps.getWriters(*maybeBuffer);
  if (!writers)
    return;
  for (OpOperand *writer : *writers) {
    if (!state.inputFusionWrites.insert(writer).second) {
      continue; // No infinite recursion
    }
    auto writeOp = dyn_cast<MemoryEffectOpInterface>(writer->getOwner());
    if (!writeOp)
      continue;
    SmallVector<MemoryEffects::EffectInstance> effects;
    writeOp.getEffects(effects);
    for (const auto &effect : effects) {
      OpOperand *maybeRecursiveReader = effect.getEffectValue<OpOperand *>();
      // Test against `writer` to guard against [MemRead, MemWrite]
      if (maybeRecursiveReader && maybeRecursiveReader != writer &&
          isa<MemoryEffects::Read>(effect.getEffect())) {
        collectInputFusionWrites(maybeRecursiveReader, bufferDeps, state);
      }
    }
  }
}

/// Given the operand that writes to a buffer, collect the writes of its reader,
/// and the writes of that reader, and so on until you're no longer writing to a
/// buffer. If any auxiliary inputs come from buffers themselves, collect
/// their writes as input fusions, because, in a fusion context, those other
/// writers will be tiled by a threadwise_read_into, not a threadwise_write_all.
static void
collectOutputFusionWrites(OpOperand *writer,
                          const BufferDependencyAnalysis &bufferDeps,
                          TransformPushState &state) {
  std::optional<memref::AllocOp> maybeBuffer =
      bufferDeps.getWrittenBuffer(writer);
  // This reads an argument or some such thing
  if (!maybeBuffer)
    return;
  if (!state.outputFusionWrites.insert(writer).second)
    return; // Prevent recursion
  std::optional<SmallVector<OpOperand *>> readers =
      bufferDeps.getReaders(*maybeBuffer);
  if (!readers)
    return;
  SmallPtrSet<OpOperand *, 4> elementwiseArgs, onwardWrites;
  for (OpOperand *reader : *readers) {
    auto readOp = dyn_cast<MemoryEffectOpInterface>(reader->getOwner());
    if (!readOp)
      continue;
    SmallVector<MemoryEffects::EffectInstance> effects;
    readOp.getEffects(effects);
    for (const auto &effect : effects) {
      OpOperand *operand = effect.getEffectValue<OpOperand *>();
      if (!operand || operand == reader)
        continue;
      if (isa<MemoryEffects::Write>(effect.getEffect())) {
        onwardWrites.insert(operand);
        elementwiseArgs.erase(operand);
      }
      if (isa<MemoryEffects::Read>(effect.getEffect()) &&
          !onwardWrites.contains(operand)) {
        elementwiseArgs.insert(operand);
      }
    }
  }
  for (OpOperand *arg : elementwiseArgs)
    collectInputFusionWrites(arg, bufferDeps, state);
  for (OpOperand *localWrite : onwardWrites) {
    LLVM_DEBUG(llvm::dbgs() << "Traversing via onward writer: "
                            << *localWrite->getOwner() << "\n");
    collectOutputFusionWrites(localWrite, bufferDeps, state);
  }
}

////////////////////////////////////////////////////////////////////////
////  Push Transforms Over alloc to fusor
////////////////////////////////////////////////////////////////////////

/// In the below, a fusor is the operation that will have some computation fused
/// into it. This is usually a gemm (either its output or inputs) but can be,
/// for example, another linalg.generic. The operation that will be fused into
/// the fusor is the fusee.
///
/// A fusor writes to or reads from a memref.alloc() and a fusee does the
/// opposite of what the fusor does. Before this pass runs, both the fusor and
/// the fusee may access that intermediate buffer (which will be rewritten away
/// by fusion) through some sequence of view operations.
///
/// After these rewrites, we have the following invariants, which make the
/// fusion code itself (it's down in AlignTiling.cpp) much simpler and enables
/// analysis of vectorization within the parts of the pipeline between this pass
/// and AlignTiling (which can be important for key performance queries).
///
/// - The fusee always has a direct reference to the buffer
///
/// Furthermore, if multiple fusees have such a non-direct reference, the
/// intermediate buffer will be duplicated so that each transform stack can be
/// inverted in isolation.

/// Invert the transform stack between `operand` and `oldAlloc`, replacing all
/// users but that transform stack with a new allocation whose type is the type
/// of the front of the transform stack. If the original transforms (and
/// allocation) then have no other uses, erase them after doing this. This also
/// updates worklists with the new allocation and reanalyzes the buffer
/// dependencies.
static LogicalResult
regularizeTransformStack(memref::AllocOp oldAlloc, OpOperand *transformedUse,
                         BufferDependencyAnalysis &bufferDeps,
                         TransformPushState &state, IRRewriter &rw) {
  SmallVector<TransformOp> transforms;
  Value transformed = transformedUse->get();
  Value buffer;
  std::tie(buffer, std::ignore) = untransform(transformed, transforms);
  if (buffer != oldAlloc.getResult()) {
    LLVM_DEBUG(llvm::dbgs()
               << "While processing " << transformed << " wanted to reach "
               << oldAlloc.getResult() << " but reached " << buffer << "\n");
    return oldAlloc->emitError(
        "mismatch between expected buffer and result of untransform() while "
        "inverting fusion transforms");
  }

  SmallVector<std::pair<TransformMapAttr, Location>> inverses;
  inverses.reserve(transforms.size());
  for (TransformOp transform : llvm::reverse(transforms)) {
    Location loc = transform.getLoc();
    TransformMapAttr inverse =
        invertTransformMap(rw, transform.getTransform(), loc);
    if (!inverse)
      return transform.emitOpError("could not invert fusee transform while "
                                   "regularizing fusions. Map = ")
             << transform.getTransform();
    inverses.emplace_back(inverse, loc);
  }

  rw.setInsertionPointAfter(oldAlloc);
  auto newAlloc = rw.create<memref::AllocOp>(
      oldAlloc.getLoc(), cast<MemRefType>(transformed.getType()));
  Value newBuffer = newAlloc.getResult();
  rw.modifyOpInPlace(transformedUse->getOwner(),
                     [&]() { transformedUse->set(newBuffer); });
  Value viewed = newBuffer;
  for (auto [inverse, loc] : llvm::reverse(inverses))
    viewed = rw.create<rock::TransformOp>(loc, viewed, inverse);
  // Don't replace the use that kicked all this off, we're probably about to
  // erase it.
  rw.replaceAllUsesExcept(buffer, viewed, transforms.back());

  for (TransformOp transform : transforms)
    if (transform.use_empty())
      rw.eraseOp(transform);
  if (oldAlloc.use_empty())
    rw.eraseOp(oldAlloc);
  else
    bufferDeps.analyze(oldAlloc);
  // If the inversions are being processed correctly, this should hit the early
  // exit case.
  bufferDeps.analyze(newAlloc);
  state.worklist.push_back(newAlloc);
  return success();
}

/// If the writer of the buffer that'll be input-fused views that buffer through
/// a series of transforms, invert those transforms and add them to the readers'
/// (fusors) transform stacks, allocating a new buffer whose type is the type of
/// that transformed view. This means that we don't need to compute inverses
/// during, say, vectorization queries in GridwiseGemmToBlockwise.
static LogicalResult pushTransformsToInputFusionReaders(
    memref::AllocOp allocOp, BufferDependencyAnalysis &bufferDeps,
    OpOperand *writer, TransformPushState &state, IRRewriter &rw) {
  Value maybeTransformedBuffer = writer->get();
  // If the input fusion writes directly to its buffer, there's nothing to do
  // here, we've met the invariant.
  if (maybeTransformedBuffer == allocOp.getResult())
    return success();
  LLVM_DEBUG(
      llvm::dbgs() << "Processing input fusion write of " << writer->get()
                   << " (argument " << writer->getOperandNumber() << " of "
                   << *writer->getOwner() << ") via " << allocOp << "\n");
  return regularizeTransformStack(allocOp, writer, bufferDeps, state, rw);
}

/// Isolate multiple readers that use transforms to view a buffer from each
/// other so that we can regularize. This function will create a copy of
/// `allocOp` for each of its readers that applies coordinate transformations
/// while it reads from `allocOp` as a fusion input. This will allow register
/// tiles to be reused in multiple contexts with multiple indexing schemes. Note
/// that the extra copies implied by this operation will be elided - in the
/// worst case, mem2reg will get through them, assuming we don't delete them.
/// (An example of such multiple uses is returning both the gemm and the sum of
/// each row within the gemm).
static LogicalResult
isolateMultipleTransformingReaders(memref::AllocOp allocOp, Operation *writer,
                                   BufferDependencyAnalysis &bufferDeps,
                                   ArrayRef<OpOperand *> readers,
                                   TransformPushState &state, IRRewriter &rw) {
  TypedValue<MemRefType> buffer = allocOp.getResult();
  Location loc = allocOp.getLoc();

  LLVM_DEBUG(llvm::dbgs() << "Fixing multiple transformed readers of "
                          << allocOp << "\n");
  for (OpOperand *reader : llvm::make_early_inc_range(readers)) {
    LLVM_DEBUG(llvm::dbgs() << "Reading op is " << *reader->getOwner() << "\n");
    Value transformed = reader->get();
    rw.setInsertionPointAfter(allocOp);
    // If there are partially-merged transform chains, isolate the transform
    // chain and update the reader.
    Value isolated = isolateTransforms(rw, transformed);
    if (isolated != transformed) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Isolated " << transformed << " to " << isolated << "\n");
      rw.modifyOpInPlace(reader->getOwner(), [&]() { reader->set(isolated); });
    }

    Value probablyBuffer;
    SmallVector<TransformOp> transforms;
    std::tie(probablyBuffer, std::ignore) = untransform(isolated, transforms);
    if (probablyBuffer != buffer)
      return allocOp.emitOpError("expected transformed reader ")
             << *(reader->getOwner()) << " to trace value to " << buffer
             << " but it reached " << probablyBuffer;
    auto newBuffer = rw.create<memref::AllocOp>(loc, buffer.getType());
    if (!transforms.empty()) {
      TransformOp viewRoot = transforms.back();
      rw.modifyOpInPlace(viewRoot, [&]() {
        viewRoot.getInputMutable().set(newBuffer.getResult());
      });
    } else {
      rw.modifyOpInPlace(reader->getOwner(),
                         [&]() { reader->set(newBuffer.getResult()); });
    }
    rw.setInsertionPointAfter(writer);
    auto copy = rw.create<memref::CopyOp>(loc, allocOp, newBuffer);
    state.worklist.push_back(newBuffer);
    state.outputFusionWrites.insert(&copy.getTargetMutable());
    bufferDeps.analyze(newBuffer);
  }

  // Prevent analysis of original allocation from going stale.
  // If this function isn't correctly dispatching all the
  // multiple-transforming-readers cases, the worklist emplacement will cause
  // infinite recursion.
  bufferDeps.analyze(allocOp);
  state.worklist.push_back(allocOp);
  return success();
}

/// Push transforms up from fusees to fusors.
///
/// This function handles the setup and the output fusion case. In this case,
/// if any of the readers of an intermediate buffer take a transformed view of
/// that buffer, we invert that view and put those inverse views on the writer
/// so that we don't need to, for example, compute inverses during vectorization
/// queries or -rock-align-tiling. After the inverses are computed, we swap in a
/// new allocation whose type is the type that the reader/fusee expects.
///
/// When there are multiple competing transform chains to invert, we introduce
/// copy operations that preserve the original buffer type so that we can invert
/// each chain onto the write of the copy and then process that.
static LogicalResult pushTransformsUp(memref::AllocOp allocOp,
                                      BufferDependencyAnalysis &bufferDeps,
                                      TransformPushState &state,
                                      IRRewriter &rw) {
  auto maybeBufferWriters = bufferDeps.getWriters(allocOp);
  auto maybeBufferReaders = bufferDeps.getReaders(allocOp);
  if (!maybeBufferWriters)
    return allocOp.emitOpError(
        "couldn't find an operation that writes this buffer");
  if (!maybeBufferReaders)
    return allocOp.emitError(
        "couldn't find an operation that reads this buffer");
  SmallVector<OpOperand *> bufferWriters = std::move(*maybeBufferWriters);
  SmallVector<OpOperand *> bufferReaders = std::move(*maybeBufferReaders);
  if (bufferWriters.size() != 1) {
    allocOp->getParentOp()->dump();
    return allocOp.emitError(
               "expected one operation to write this buffer, but there are ")
           << bufferWriters.size();
  }

  OpOperand *writer = bufferWriters[0];
  bool isInputFusion = state.inputFusionWrites.contains(writer);
  if (isInputFusion)
    return pushTransformsToInputFusionReaders(allocOp, bufferDeps, writer,
                                              state, rw);
  if (!state.outputFusionWrites.contains(writer)) {
    return allocOp.emitOpError("couldn't find the writer of a buffer in the "
                               "set of analyzed fusable writes. See ")
           << bufferWriters.front()->get();
  }

  OpOperand *transformingReader = nullptr;
  for (OpOperand *reader : bufferReaders) {
    Value readArg = reader->get();
    if (!readArg.getDefiningOp<rock::TransformOp>()) {
      continue;
    }
    if (transformingReader && transformingReader != reader)
      return isolateMultipleTransformingReaders(
          allocOp, writer->getOwner(), bufferDeps, bufferReaders, state, rw);
    transformingReader = reader;
  }
  // It's the simple case, do nothing
  if (!transformingReader)
    return success();

  if (transformingReader && bufferReaders.size() > 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "Adding copies to partially intercept a transform stack\n");
    return isolateMultipleTransformingReaders(
        allocOp, writer->getOwner(), bufferDeps, bufferReaders, state, rw);
  }

  LLVM_DEBUG(llvm::dbgs() << "Processing output fusion read via "
                          << transformingReader->get() << " (argument "
                          << transformingReader->getOperandNumber() << " of "
                          << *transformingReader->getOwner() << ") via "
                          << allocOp << "\n");

  return regularizeTransformStack(allocOp, transformingReader, bufferDeps,
                                  state, rw);
}

LogicalResult findFusionRoots(func::FuncOp kernel,
                              const BufferDependencyAnalysis &bufferDeps,
                              TransformPushState &state) {
  bool foundFusionRoot = false;
  kernel.walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case(
            [&](memref::AllocOp allocOp) { state.worklist.push_back(allocOp); })
        .Case([&](GridwiseGemmOp gemmOp) {
          foundFusionRoot = true;
          collectOutputFusionWrites(&gemmOp.getCMutable(), bufferDeps, state);
          collectInputFusionWrites(&gemmOp.getAMutable(), bufferDeps, state);
          collectInputFusionWrites(&gemmOp.getBMutable(), bufferDeps, state);
        })
        .Case([&](GridwiseGemmAccelOp gemmOp) {
          foundFusionRoot = true;
          collectOutputFusionWrites(&gemmOp.getCMutable(), bufferDeps, state);
          collectInputFusionWrites(&gemmOp.getAMutable(), bufferDeps, state);
          collectInputFusionWrites(&gemmOp.getBMutable(), bufferDeps, state);
        })
        .Case([&](GridwiseAttentionAccelOp attenOp) {
          foundFusionRoot = true;
          collectOutputFusionWrites(&attenOp.getOutMutable(), bufferDeps,
                                    state);
          collectInputFusionWrites(&attenOp.getQueriesMutable(), bufferDeps,
                                   state);
          collectInputFusionWrites(&attenOp.getKeysMutable(), bufferDeps,
                                   state);
          collectInputFusionWrites(&attenOp.getValuesMutable(), bufferDeps,
                                   state);

          // The linalg.generic inside the attention's body will be expected to
          // write out a global tensor as if it were an output fusion, so its
          // write should be added to the set of output fusion writes lest we
          // get complaints that there's an alloc that doesn't participatie in
          // fusion.
          for (auto lgop :
               attenOp.getPreSoftmaxBody().getOps<linalg::GenericOp>()) {
            collectOutputFusionWrites(&lgop.getOutputsMutable()[0], bufferDeps,
                                      state);
          }
        });
  });
  // Make the traversal top-down just to be safe.
  std::reverse(state.worklist.begin(), state.worklist.end());

  return success(foundFusionRoot);
}

static LogicalResult runPushTransformsUp(func::FuncOp op,
                                         BufferDependencyAnalysis &bufferDeps) {
  TransformPushState state;
  if (failed(findFusionRoots(op, bufferDeps, state))) {
    // This pass is being run post gridwise-gemm-to-blockwise or otherwise out
    // of its context, and so should do nothing. Usually this is because the
    // early-pipeline applicability passes are being re-run as part of the
    // lowering pipeline.
    return success();
  }
  IRRewriter rw(op.getContext());
  while (!state.worklist.empty()) {
    memref::AllocOp allocOp = state.worklist.pop_back_val();
    LLVM_DEBUG(llvm::dbgs() << "Regularizing: " << allocOp << "\n");
    if (failed(pushTransformsUp(allocOp, bufferDeps, state, rw)))
      return failure();
    LLVM_DEBUG(llvm::dbgs() << "// --- //\n");
  }
  return success();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
struct RockRegularizePass
    : public rock::impl::RockRegularizePassBase<RockRegularizePass> {
  void runOnOperation() override;
};
} // end namespace

void RockRegularizePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto func = getOperation();
  if (!func->hasAttr("kernel")) {
    // disable for non-kernels
    return;
  }

  {
    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                           memref::MemRefDialect, linalg::LinalgDialect>();
    target.addIllegalOp<memref::ExpandShapeOp, memref::CollapseShapeOp>();
    target.addDynamicallyLegalOp<linalg::GenericOp>(
        [](linalg::GenericOp op) { return isRegularGeneric(op); });

    RewritePatternSet patterns(ctx);
    patterns.add<CollapseRewritePattern, ExpandRewritePattern,
                 RegularizeGenericRewritePattern>(ctx);
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  {
    auto &bufferDeps = getAnalysis<BufferDependencyAnalysis>();
    if (failed(runPushTransformsUp(func, bufferDeps)))
      return signalPassFailure();
  }

  func->walk(annotateGenericOp);
}
