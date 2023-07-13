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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

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

struct LAGenericRewritePattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp laGeneric,
                                PatternRewriter &b) const override;
};

struct MemcpyRewritePattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &b) const override;
};

struct ReduceRewritePattern : public OpRewritePattern<rock::ReduceOp> {
  using OpRewritePattern<rock::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override;
};
} // end anonymous namespace

static void moveTransformsBefore(PatternRewriter &b, Value val) {
  if (auto defOp = val.getDefiningOp()) {
    if (auto rtop = dyn_cast<TransformOp>(defOp)) {
      moveTransformsBefore(b, rtop.getOperand());

      defOp->remove();
      b.insert(defOp);
    } else {
      llvm_unreachable("must trace to func.arg");
    }
  }
}

static Value applyViewsOnDest(PatternRewriter &rewriter, Location loc,
                              Value dest, ArrayRef<TransformMapAttr> views) {
  for (TransformMapAttr trMap : views) {
    dest = rewriter.create<TransformOp>(loc, dest, trMap);
  }
  return dest;
}

Value makeRegs(PatternRewriter &b, MemRefType::Builder &mrb, Location loc,
               Type srcType) {
  auto srcMemType = srcType.cast<MemRefType>();
  // 1. create a second allocation of the same type to hold loaded elements
  return b.create<GpuAllocOp>(loc, static_cast<MemRefType>(mrb.setElementType(
                                       srcMemType.getElementType())));
}

static Value applyTransforms(PatternRewriter &b, ThreadwiseWriteAllOp storeOp,
                             Value src,
                             ArrayRef<TransformMapAttr> relativeViewsOnStore) {
  // 0. capture the memref containing the outputs being written
  Location loc = storeOp.getLoc();
  Value gemmOut = storeOp.getSource();

  // 1. create a second allocation of the same type to hold loaded elements
  MemRefType::Builder mrb(gemmOut.getType().cast<MemRefType>());
  Value alloc = makeRegs(b, mrb, loc, src.getType());

  // 2. clone twcopy for <addend> into regs
  LLVM_DEBUG(llvm::dbgs() << "Src type: " << src.getType() << " dest type: "
                          << storeOp.getDest().getType() << "\n");

  // 2.0. first move all transforms before the relocated linalg.generic
  moveTransformsBefore(b, src);

  // 2.1. apply transform chain from output
  src = applyViewsOnDest(b, loc, src, relativeViewsOnStore);

  // 2.2. load into registers
  Value bid = b.createOrFold<rock::WorkgroupIdOp>(loc, b.getIndexType());
  Value tid = b.createOrFold<rock::WorkitemIdOp>(loc, b.getIndexType());
  b.create<ThreadwiseReadIntoOp>(loc, src, alloc, storeOp.getExtraViews(),
                                 /*extraIndices=*/ValueRange{bid, tid},
                                 storeOp.getForceUnroll(),
                                 storeOp.getUseIndexDiffs());
  return alloc;
}

static void traceToNonViewUsers(Operation *op,
                                SmallVectorImpl<Operation *> &nonViewOps) {
  if (auto transform = dyn_cast<TransformOp>(op)) {
    Value result = transform.getResult();
    for (auto &use : result.getUses()) {
      traceToNonViewUsers(use.getOwner(), nonViewOps);
    }
  } else {
    nonViewOps.push_back(op);
  }
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
    for (auto &use : result.getUses()) {
      return traceToNonViewReaders(use.getOwner(), result, nonViewReaders);
    }
  } else {
    if (ThreadwiseWriteAllOp twWriteAllOp =
            dyn_cast<ThreadwiseWriteAllOp>(op)) {
      if (twWriteAllOp.getSource() == parentVal) {
        nonViewReaders.push_back(op);
      }
    } else if (MemoryEffectOpInterface memEffectOp =
                   dyn_cast<MemoryEffectOpInterface>(op)) {
      if (hasEffect<MemoryEffects::Read>(memEffectOp, parentVal)) {
        nonViewReaders.push_back(op);
      }
    } else if (CopyOpInterface copyOp = dyn_cast<CopyOpInterface>(op)) {
      if (copyOp.getSource() == parentVal) {
        nonViewReaders.push_back(op);
      }
    } else {
      return op->emitError() << "Found an unsupported operator that needs to "
                                "be added reader checks \n"
                             << op;
    }
  }
  return success();
}

static Operation *traceToNonViewDef(Operation *op,
                                    SmallVectorImpl<TransformMapAttr> &views,
                                    Operation *terminator) {
  if (op == terminator) {
    return op;
  }
  if (auto transform = dyn_cast<TransformOp>(op)) {
    Operation *nonViewDef = traceToNonViewDef(
        transform.getViewSource().getDefiningOp(), views, terminator);
    views.push_back(transform.getTransformAttr());
    return nonViewDef;
  }
  return op;
}

static Operation *traceToNonViewDef(Operation *op) {
  if (auto transform = dyn_cast<TransformOp>(op)) {
    return traceToNonViewDef(transform.getViewSource().getDefiningOp());
  } else {
    return op;
  }
}

// This function checks given a parent operation and a reader candidate
// whether that reader is the unique reader of the parent operation.
// This is to be used identify where interrim memrefs could be eliminated
// and fused.
static LogicalResult checkUniqueReader(Operation *op, Operation *reader,
                                       bool &isUnique) {
  op = traceToNonViewDef(op);
  SmallVector<Operation *, 4> nonViewReaders;
  Operation *nonViewDef = traceToNonViewDef(op);
  for (Value result : nonViewDef->getResults()) {
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
  if (nonViewReaders.size() != 1) {
    isUnique = false;
  }
  isUnique = (nonViewReaders[0] == reader);
  return success();
}

static ThreadwiseWriteAllOp traceToThreadwiseWrite(
    Value inp, SmallVectorImpl<TransformMapAttr> &inpToTWWriteAllViews) {
  // 1. Validate that the only uses of the linalg.generic input are the one
  // generic and a copy operation or transform.
  ThreadwiseWriteAllOp result;
  for (Operation *use : inp.getUsers()) {
    SmallVector<Operation *, 4> nonViewUsers;
    traceToNonViewUsers(use, nonViewUsers);
    for (Operation *nonViewUse : nonViewUsers) {
      if (isa<memref::DeallocOp>(nonViewUse)) {
        // ignore
        continue;
      }
      if (auto lgop = dyn_cast<linalg::GenericOp>(nonViewUse)) {
        // We dont check for linalg.generic readers
        // here because they could both be reading and writing
        // to intermediary buffer that could be copied as an
        // additional kernel output.
      } else if (auto reduceOp = dyn_cast<ReduceOp>(nonViewUse)) {
        // reader
        if (inp != reduceOp.getIn()) {
          return ThreadwiseWriteAllOp();
        }
      } else if (auto memcpy = dyn_cast<memref::CopyOp>(nonViewUse)) {
        // reader
        if (memcpy.getSource() != inp) {
          return ThreadwiseWriteAllOp();
        }
      } else if (auto store = dyn_cast<ThreadwiseWriteAllOp>(nonViewUse)) {
        // Threadwise copy that is already unttransformed (new style)
        if (result) {
          LLVM_DEBUG(llvm::dbgs() << "Multiple global stores somehow\n");
          return ThreadwiseWriteAllOp();
        } else {
          SmallVector<TransformMapAttr> views;
          traceToNonViewDef(store.getDest().getDefiningOp(), views,
                            inp.getDefiningOp());
          result = store;
          inpToTWWriteAllViews.insert(inpToTWWriteAllViews.begin(),
                                      views.begin(), views.end());
        }
      } else {
        return ThreadwiseWriteAllOp();
      }
    }
  }

  if (!result)
    LLVM_DEBUG(llvm::dbgs() << "Align tiling: generic not tracing to copy\n");
  return result;
}

// Returns the value of the buffer that's meant to be the new writeback.
static void
reconfigureLAGeneric(PatternRewriter &b, linalg::GenericOp laGeneric,
                     Value inRegs, Value outRegs,
                     ThreadwiseWriteAllOp twWriteOp,
                     ArrayRef<TransformMapAttr> relativeViewsOnStore) {
  SmallVector<Value, 5> newInputs;
  SmallVector<AffineMap, 5> lgAMaps;

  for (auto inp : laGeneric.getInputs()) {
    Value newInput;
    SmallVector<TransformMapAttr> views;
    if (traceToThreadwiseWrite(inp, views)) {
      newInput = inRegs;
    } else {
      // 2.1.1. Align tiling of other inputs
      newInput = applyTransforms(b, twWriteOp, inp, relativeViewsOnStore);
    }
    newInputs.push_back(newInput);

    auto inpRank = newInput.getType().cast<ShapedType>().getRank();
    lgAMaps.push_back(b.getMultiDimIdentityMap(inpRank));
  }

  laGeneric.getInputsMutable().assign(newInputs);
  laGeneric.getOutputsMutable().assign(outRegs);

  // 2.2. Reset affine maps
  auto regRank = inRegs.getType().cast<ShapedType>().getRank();

  lgAMaps.push_back(b.getMultiDimIdentityMap(regRank));
  laGeneric.setIndexingMapsAttr(b.getAffineMapArrayAttr(lgAMaps));

  // 2.3. Reset iterator types
  MLIRContext *ctx = b.getContext();
  SmallVector<Attribute, 5> iteratorTypes;
  iteratorTypes.resize(regRank, linalg::IteratorTypeAttr::get(
                                    ctx, utils::IteratorType::parallel));
  laGeneric.setIteratorTypesAttr(ArrayAttr::get(ctx, iteratorTypes));
}

static Value
findThreadwiseWrite(linalg::GenericOp laGeneric,
                    ThreadwiseWriteAllOp &twWriteOp,
                    SmallVector<TransformMapAttr> &laInToOutViews) {
  for (auto input : laGeneric.getInputs()) {
    if (auto twop = traceToThreadwiseWrite(input, laInToOutViews)) {
      twWriteOp = twop;
      return input;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "No input is leading to a global store.\n");
  return Value();
}

LogicalResult
LAGenericRewritePattern::matchAndRewrite(linalg::GenericOp laGeneric,
                                         PatternRewriter &b) const {
  Location loc = laGeneric.getLoc();

  // 0. Test compatibility
  // 0.0. Only fully parallel for now
  for (utils::IteratorType iterType : laGeneric.getIteratorTypesArray())
    if (!linalg::isParallelIterator(iterType))
      return laGeneric.emitError("must be fully parallel");

  // 0.1. Test compatibility,  Only 1 output supported
  if (laGeneric.getOutputs().size() != 1)
    return laGeneric.emitError("only 1 output supported");
  Value out = *laGeneric.getOutputs().begin();

  // 0.2. Sanity check, skip already fused.
  for (auto inp : laGeneric.getInputs()) {
    if (auto fusedAlloc = inp.getDefiningOp<GpuAllocOp>()) {
      LLVM_DEBUG(llvm::dbgs() << "Found existing fusion, bailing\n");
      return failure();
    }
  }

  // 1. Trace input to global_store.
  // 1.1. Find the (implicit) gemm output
  ThreadwiseWriteAllOp gemmStoreOp;
  SmallVector<TransformMapAttr> laInToOutViews;
  Value laGenericInputLeadingToGemmStore =
      findThreadwiseWrite(laGeneric, gemmStoreOp, laInToOutViews);
  if (!laGenericInputLeadingToGemmStore)
    return failure();

  auto outType = out.getType().cast<ShapedType>();
  auto inpType = laGenericInputLeadingToGemmStore.getType().cast<ShapedType>();
  if (outType.getShape() != inpType.getShape()) {
    return laGeneric.emitError("input and output types must match");
  }

  for (auto idxMap : laGeneric.getIndexingMapsArray()) {
    if (!idxMap.isIdentity()) {
      return laGeneric.emitError("indexing_maps must all be identity");
    }
  }

  if (!gemmStoreOp) {
    LLVM_DEBUG(llvm::dbgs() << "Align tiling: couldn't find writeback\n");
    return failure();
  }

  // We check the input leading to GEMM store has the current LA generic,
  // that is being rewritten, as the unique reader. This is because if it is the
  // unique reader, the previous memref does not need to be maintained anymore
  // and we can directly write into the outputs of the current linalg blocks.
  // Moreover, this is checked before any rewrite happens to avoid interference
  // and used later below.
  bool isUniqueReader;
  LogicalResult checkResult =
      checkUniqueReader(laGenericInputLeadingToGemmStore.getDefiningOp(),
                        laGeneric, isUniqueReader);
  if (checkResult.failed()) {
    return checkResult;
  }

  // 2. Apply if input found
  Value gemmOut = gemmStoreOp.getSource();
  auto gemmOutType = gemmOut.getType().cast<MemRefType>();

  Value fusionRegs;
  {
    PatternRewriter::InsertionGuard guard(b);
    // 2.0. Reset insertion point to before the copy.
    b.setInsertionPoint(gemmOut.getDefiningOp());
    // 2.1. Take out a slice of the result vector to create a vector-sized
    // slice to enable creating the fusion section.
    fusionRegs = b.create<GpuAllocOp>(loc, gemmOutType);
  }

  PatternRewriter::InsertionGuard guard(b);
  // 2.0. Reset insertion point to before the copy.
  b.setInsertionPoint(gemmStoreOp);

  MemRefType::Builder mrb(gemmOutType);
  auto laOutRegs = makeRegs(b, mrb, loc, out.getType());
  // 2.2. Tile linalg.generic with vgpr as input, return output vgprs
  reconfigureLAGeneric(b, laGeneric, fusionRegs, laOutRegs, gemmStoreOp,
                       laInToOutViews);
  // 2.2.0. Move the generic before the write-back. This'll put all
  // the copy loops for other inputs before the generic due to insertion
  // order.

  // We need to clone ThreadwiseWriteAllOp if the current op is not the unique
  // reader of it because the fusion process will eliminate the input memref.
  if (!isUniqueReader) {
    gemmStoreOp =
        static_cast<ThreadwiseWriteAllOp>(b.clone(*gemmStoreOp.getOperation()));
  }
  gemmStoreOp->moveAfter(laGeneric);
  gemmOut.replaceAllUsesWith(fusionRegs);

  // 2.3. Replace rock.threadwise_write_all inputs with la.generic result vgprs
  gemmStoreOp.getSourceMutable().assign(laOutRegs);
  out = applyViewsOnDest(b, loc, out, laInToOutViews);
  gemmStoreOp.getDestMutable().assign(out);

  if (auto outAlloc = out.getDefiningOp<memref::AllocOp>()) {
    outAlloc->moveBefore(gemmStoreOp);
  }

  return success();
}

LogicalResult MemcpyRewritePattern::matchAndRewrite(memref::CopyOp copy,
                                                    PatternRewriter &b) const {
  auto src = copy.getSource();
  auto target = copy.getTarget();
  Location loc = copy.getLoc();

  Operation *gemmStoreOp = nullptr;
  SmallVector<TransformMapAttr> views;
  if (auto allocOp = src.getDefiningOp<memref::AllocOp>()) {
    if (auto twop = traceToThreadwiseWrite(src, views)) {
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
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPoint(twWriteAllOp);

      // 1. replace memref.copy with rock.threadwise_write_all
      target = cast<TypedValue<BaseMemRefType>>(
          applyViewsOnDest(b, loc, target, views));
      twWriteAllOp.getDestMutable().assign(target);
      // twWriteAllOp->moveAfter(target.getDefiningOp());

      if (auto outAlloc = target.getDefiningOp<memref::AllocOp>()) {
        outAlloc->moveBefore(gemmStoreOp);
      }

      b.eraseOp(copy);
      return success();
    }
  }

  return failure();
}

static bool isUnfusedKernelStore(ThreadwiseWriteAllOp store) {
  bool ret = isa_and_nonnull<memref::AllocOp>(store.getDest().getDefiningOp());
  if (ret) {
    store.getDest().getDefiningOp()->emitOpError(
        "could not use fusion to eliminate this intermediate buffer. Kernel "
        "compilation cannot proceed");
  }
  return ret;
}

LogicalResult
ReduceRewritePattern::matchAndRewrite(rock::ReduceOp reduceOp,
                                      PatternRewriter &rewriter) const {
  Location loc = reduceOp.getLoc();
  SmallVector<TransformMapAttr, 4> views;
  ThreadwiseWriteAllOp threadwiseWriteOp =
      traceToThreadwiseWrite(reduceOp.getIn(), views);
  if (!threadwiseWriteOp) {
    return rewriter.notifyMatchFailure(
        reduceOp, "rock.reduce input is not leading to ThreadwiseWriteAllOp of "
                  "the previous op.");
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
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(threadwiseWriteOp);
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
  {
    RewritePatternSet patterns(ctx);
    patterns.add<LAGenericRewritePattern>(ctx);
    patterns.add<ReduceRewritePattern>(ctx);
    patterns.add<MemcpyRewritePattern>(ctx);
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config)))
      signalPassFailure();
  }

  {
    WalkResult verifyAllStores =
        getOperation().walk([](ThreadwiseWriteAllOp store) {
          return isUnfusedKernelStore(store) ? WalkResult::interrupt()
                                             : WalkResult::advance();
        });
    if (verifyAllStores.wasInterrupted())
      signalPassFailure();
  }
}
