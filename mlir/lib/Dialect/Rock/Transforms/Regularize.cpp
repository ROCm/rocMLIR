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
                  IntegerAttr::get(IndexType::get(ctx), 0));
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

/// The overall goal of this pass is to make sore you can go from the operands
/// of the fusion root (usually a GEMM or attention) to function arguments
/// through any intervening fusable operations (generally linalg.generic
/// operations or reductions) without needing to invert any transformations.
/// This allows us to accurately (assuming the heuristic for multiple-output
/// input fusions isn't meaningfully incorrect) analyze the memory layout of the
/// underlying memory despite any intervening fusion code. This is important,
/// especially in the case of input fusions, as we rely on those vectorization
/// analyses in order to correctly select elements of our schedule, like whether
/// to do a bank conflict avoidance swizzle.
///
/// To achaive this, we traverse the writeOperands starting from the output
/// buffer of each "fusion root" (the C algument of GEMM, the output of
/// attention, etc) and record all the buffer-writing operands on the way from
/// that argument to a function result (such as the outputs of an intervening
/// linalg.generic) and classify them as output fusion writes. That is, output
/// fusion writes are all those that will be replaced by writes to a register
/// buffer during fusion. We then classify all the buffer writes between
/// funciton inputs and the read operands of a fusion root as input fusion
/// writes, along with the writes needed to produce any elementwise arguments to
/// operations in the output fusion path (ex. if the bias elements are
/// constructed by a linalg.generic, the output operand of that generic is an
/// input fusion write). Equivalently, input fusion writes are those that become
/// reads into a register buffer after fusion.
///
/// Once we have done this, we regularize the function as follows:
/// - If the operand that points at a buffer write is on the output path,
/// all the readers of that buffer must refer to that buffer directly and not
/// through transforms. If there are such transforms, we invert them and replace
/// the buffer with one whose type is the one expected by the reading operation.
/// Shoulld there be multiple such operations that disagree about what the type
/// of that buffer should be, we copy the original output buffer so that the
/// inversions can be performed independently.
/// - Conversely, for operands on the input path, the operation that writes
/// to the buffer has to refer to the bufffer directly, so any views on that
/// writing operand are inverted and those inverses become part of the view
/// stack for each operation reading the buffer. This lets us traverse from
/// the writing operand to its input during a vectorization analysis without
/// having to invert the maps on the output.

namespace {
/// Information needed while regularizing the view chains to allow vectorization
/// and permit fusion.
/// `worklist` is the set of memref.alloc ops that haven't been processed yet.
/// When we do a non-trivial regularization, we always add the new buffer
/// to the worklist to ensure that the regularization invariants are mantained.
/// `inputFusionWrites` is the set of `OpOperand*`s that point at arguments
/// of operations that write to a buffer where the underlying operation is on
/// an input fusion chain.
/// `outputFusionWrites` is similarly the set of operands indicating a
/// potentially transformed buffer that's written during the main output fusion
/// chain. These sets are needed to resolve the ambiguity abut how to regularize
/// a `linalg.generic` - that is, can its `out` have a view on it or not? Note
/// that this is mutable state because, in the case of conflicting
/// transformations in an output fusion, we'll need to copy the intermediate
/// buffer so we can invert the conflicting transformations onto the writes
/// of each copy.
struct TransformPushState {
  llvm::SmallVector<memref::AllocOp> worklist;
  llvm::SmallPtrSet<OpOperand *, 8> inputFusionWrites;
  llvm::SmallPtrSet<OpOperand *, 8> outputFusionWrites;
};
} // end namespace

/// Given an operand that reads a buffer, collect the write operand that filled
/// that buffer and then recurse to all the buffers that that writing operation
/// depends on. This allows us to classify linalg.generic operations as
/// belonging to input fusions.
static void
collectInputFusionWriteOperands(OpOperand *readOperand,
                                const BufferDependencyAnalysis &bufferDeps,
                                TransformPushState &state) {
  std::optional<memref::AllocOp> maybeBuffer =
      bufferDeps.getReadBuffer(readOperand);
  // This reads an argument or some such thing
  if (!maybeBuffer)
    return;
  std::optional<SmallVector<OpOperand *>> writeOperands =
      bufferDeps.getWriters(*maybeBuffer);
  if (!writeOperands)
    return;
  for (OpOperand *writeOperand : *writeOperands) {
    if (!state.inputFusionWrites.insert(writeOperand).second) {
      continue; // No infinite recursion
    }
    auto writeOp = dyn_cast<MemoryEffectOpInterface>(writeOperand->getOwner());
    if (!writeOp)
      continue;
    SmallVector<MemoryEffects::EffectInstance> effects;
    writeOp.getEffects(effects);
    for (const auto &effect : effects) {
      OpOperand *maybeRecursiveReadOperand =
          effect.getEffectValue<OpOperand *>();
      // Test against the write operand to guard against [MemRead, MemWrite]
      if (maybeRecursiveReadOperand &&
          maybeRecursiveReadOperand != writeOperand &&
          isa<MemoryEffects::Read>(effect.getEffect())) {
        collectInputFusionWriteOperands(maybeRecursiveReadOperand, bufferDeps,
                                        state);
      }
    }
  }
}

/// Given the operand that writes to a buffer, traverse all the operands that
/// are reads from that buffer and examine their operations. Then, for each
/// buffer that that reading operation accesses, if it is read, classify its
/// producer as part of the input fusion process, while if the buffer is
/// written, recurse to collect that write. This classifies the intermediate
/// buffers for the output fusion chain as those that will be read from (their
/// tile size will come from a threadwise_read_into) and those that will be
/// written (their tile size will come from a threadwise_write_all).
static void
collectOutputFusionWrites(OpOperand *writeOperand,
                          const BufferDependencyAnalysis &bufferDeps,
                          TransformPushState &state) {
  // This is, broadly, a cached untransform() except that it fails when you hit
  // a function argument, which is what we want here.
  std::optional<memref::AllocOp> maybeBuffer =
      bufferDeps.getWrittenBuffer(writeOperand);
  // This reads an argument or some such thing
  if (!maybeBuffer)
    return;
  if (!state.outputFusionWrites.insert(writeOperand).second)
    return; // Prevent recursion
  std::optional<SmallVector<OpOperand *>> readOperands =
      bufferDeps.getReaders(*maybeBuffer);
  if (!readOperands)
    return;
  SmallPtrSet<OpOperand *, 4> elementwiseArgs, onwardWrites;
  for (OpOperand *readOperand : *readOperands) {
    auto readOp = dyn_cast<MemoryEffectOpInterface>(readOperand->getOwner());
    if (!readOp)
      continue;
    SmallVector<MemoryEffects::EffectInstance> effects;
    readOp.getEffects(effects);
    for (const auto &effect : effects) {
      OpOperand *operand = effect.getEffectValue<OpOperand *>();
      if (!operand || operand == readOperand)
        continue;
      MemoryEffects::Effect *effectType = effect.getEffect();
      if (isa<MemoryEffects::Write>(effectType)) {
        onwardWrites.insert(operand);
        elementwiseArgs.erase(operand);
      }
      if (isa<MemoryEffects::Read>(effectType) &&
          !onwardWrites.contains(operand)) {
        elementwiseArgs.insert(operand);
      }
    }
  }
  for (OpOperand *arg : elementwiseArgs)
    collectInputFusionWriteOperands(arg, bufferDeps, state);
  for (OpOperand *localWrite : onwardWrites) {
    LLVM_DEBUG(llvm::dbgs() << "Traversing via onward writer: "
                            << *localWrite->getOwner() << "\n");
    collectOutputFusionWrites(localWrite, bufferDeps, state);
  }
}

////////////////////////////////////////////////////////////////////////
////  Push Transforms Over alloc to fusor
////////////////////////////////////////////////////////////////////////

/// Invert the transform stack between `operand` and `oldAlloc`, replacing all
/// users but that transform stack with a new allocation whose type is the type
/// of `transformedUse`'s value. If the original transforms (and
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
static LogicalResult isolateMultipleTransformingReaders(
    memref::AllocOp allocOp, Operation *writeOperand,
    BufferDependencyAnalysis &bufferDeps, ArrayRef<OpOperand *> readOperands,
    TransformPushState &state, IRRewriter &rw) {
  TypedValue<MemRefType> buffer = allocOp.getResult();
  Location loc = allocOp.getLoc();

  LLVM_DEBUG(llvm::dbgs() << "Fixing multiple transformed readers of "
                          << allocOp << "\n");
  for (OpOperand *readOperand : llvm::make_early_inc_range(readOperands)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Reading op is " << *readOperand->getOwner() << "\n");
    Value transformed = readOperand->get();
    rw.setInsertionPointAfter(allocOp);
    // If there are partially-merged transform chains, isolate the transform
    // chain and update the reader.
    Value isolated = isolateTransforms(rw, transformed);
    if (isolated != transformed) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Isolated " << transformed << " to " << isolated << "\n");
      rw.modifyOpInPlace(readOperand->getOwner(),
                         [&]() { readOperand->set(isolated); });
    }

    Value probablyBuffer;
    SmallVector<TransformOp> transforms;
    std::tie(probablyBuffer, std::ignore) = untransform(isolated, transforms);
    if (probablyBuffer != buffer)
      return allocOp.emitOpError("expected transformed reader ")
             << *(readOperand->getOwner()) << " to trace value to " << buffer
             << " but it reached " << probablyBuffer;
    auto newBuffer = rw.create<memref::AllocOp>(loc, buffer.getType());
    if (!transforms.empty()) {
      TransformOp viewRoot = transforms.back();
      rw.modifyOpInPlace(viewRoot, [&]() {
        viewRoot.getInputMutable().set(newBuffer.getResult());
      });
    } else {
      rw.modifyOpInPlace(readOperand->getOwner(),
                         [&]() { readOperand->set(newBuffer.getResult()); });
    }
    rw.setInsertionPointAfter(writeOperand);
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

/// Entry point for the regularization rewrite that inverts transforms.
///
/// This function dispatches to the input fusion case and the multiple
/// transforming readers case, and then handles inverting the transforms on the
/// operation argument corresponding to a read from an intermediate buffer onto
/// tho operation that writes that buffer.
static LogicalResult pushTransformsUp(memref::AllocOp allocOp,
                                      BufferDependencyAnalysis &bufferDeps,
                                      TransformPushState &state,
                                      IRRewriter &rw) {
  auto maybeBufferWriterOperands = bufferDeps.getWriters(allocOp);
  auto maybeBufferReaderOperands = bufferDeps.getReaders(allocOp);
  if (!maybeBufferWriterOperands)
    return allocOp.emitOpError(
        "couldn't find an operation that writes this buffer");
  if (!maybeBufferReaderOperands)
    return allocOp.emitError(
        "couldn't find an operation that reads this buffer");
  SmallVector<OpOperand *> writeOperands =
      std::move(*maybeBufferWriterOperands);
  SmallVector<OpOperand *> readOperands = std::move(*maybeBufferReaderOperands);
  if (writeOperands.size() != 1) {
    allocOp->getParentOp()->dump();
    return allocOp.emitError(
               "expected one operation to write this buffer, but there are ")
           << writeOperands.size();
  }

  OpOperand *writeOperand = writeOperands[0];
  bool isInputFusion = state.inputFusionWrites.contains(writeOperand);
  if (isInputFusion)
    return pushTransformsToInputFusionReaders(allocOp, bufferDeps, writeOperand,
                                              state, rw);
  if (!state.outputFusionWrites.contains(writeOperand)) {
    return allocOp.emitOpError("couldn't find the writer of a buffer in the "
                               "set of analyzed fusable writes. See ")
           << writeOperand->get();
  }

  OpOperand *readOperandWithTransforms = nullptr;
  for (OpOperand *readOperand : readOperands) {
    Value maybeTransformedBuffer = readOperand->get();
    if (!maybeTransformedBuffer.getDefiningOp<rock::TransformOp>()) {
      continue;
    }
    if (readOperandWithTransforms && readOperandWithTransforms != readOperand)
      return isolateMultipleTransformingReaders(
          allocOp, writeOperand->getOwner(), bufferDeps, readOperands, state,
          rw);
    readOperandWithTransforms = readOperand;
  }
  // It's the simple case, do nothing
  if (!readOperandWithTransforms)
    return success();

  if (readOperandWithTransforms && readOperands.size() > 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "Adding copies to partially intercept a transform stack\n");
    return isolateMultipleTransformingReaders(
        allocOp, writeOperand->getOwner(), bufferDeps, readOperands, state, rw);
  }

  LLVM_DEBUG(llvm::dbgs() << "Processing output fusion read via "
                          << readOperandWithTransforms->get() << " (argument "
                          << readOperandWithTransforms->getOperandNumber()
                          << " of " << *readOperandWithTransforms->getOwner()
                          << ") via " << allocOp << "\n");

  return regularizeTransformStack(allocOp, readOperandWithTransforms,
                                  bufferDeps, state, rw);
}

LogicalResult findFusionRoots(func::FuncOp kernel,
                              const BufferDependencyAnalysis &bufferDeps,
                              TransformPushState &state) {
  bool foundFusionRoot = false;
  kernel.walk([&](Operation *op) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      state.worklist.push_back(allocOp);
      return;
    }

    if (op->hasTrait<OpTrait::rock::FusionRoot>()) {
      foundFusionRoot = true;
      auto effectsIface = cast<MemoryEffectOpInterface>(op);
      SmallVector<MemoryEffects::EffectInstance> effects;
      effectsIface.getEffects(effects);
      SmallPtrSet<OpOperand *, 4> readOperands, writeOperands;
      for (const auto &effect : effects) {
        OpOperand *operand = effect.getEffectValue<OpOperand *>();
        if (!operand)
          continue;
        MemoryEffects::Effect *effectType = effect.getEffect();
        if (isa<MemoryEffects::Read>(effectType) &&
            !writeOperands.contains(operand)) {
          readOperands.insert(operand);
        } else if (isa<MemoryEffects::Write>(effectType)) {
          writeOperands.insert(operand);
          readOperands.erase(operand);
        }
      }
      for (OpOperand *writeOperand : writeOperands)
        collectOutputFusionWrites(writeOperand, bufferDeps, state);
      for (OpOperand *readOperand : readOperands)
        collectInputFusionWriteOperands(readOperand, bufferDeps, state);
    }

    if (isa<AttentionOp, GridwiseAttentionAccelOp>(op)) {
      // The linalg.generic inside the attention's body will be expected to
      // write out a global tensor as if it were an output fusion, so its
      // write should be added to the set of output fusion writes lest we
      // get complaints that there's an alloc that doesn't participatie in
      // fusion.
      for (Region &attachedFunc : op->getRegions()) {
        for (auto lgop : attachedFunc.getOps<linalg::GenericOp>()) {
          collectOutputFusionWrites(&lgop.getOutputsMutable()[0], bufferDeps,
                                    state);
        }
      }
    }
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
