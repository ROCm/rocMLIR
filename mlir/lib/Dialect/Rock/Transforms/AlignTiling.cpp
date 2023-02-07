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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
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

static Value transformAccordingly(PatternRewriter &b, Value nOut, Value refOp) {
  if (auto rtop = dyn_cast<TransformOp>(refOp.getDefiningOp())) {
    // 0. recurse and apply from the begining of the transform chain
    nOut = transformAccordingly(b, nOut, rtop.getOperand());

    // 1. apply identical transforms to other inputs
    BlockAndValueMapping cmap;
    cmap.map(rtop.getOperand(), nOut);
    auto ntop = dyn_cast<TransformOp>(b.clone(*rtop, cmap));
    nOut = ntop.getResult();
  }

  return nOut;
}

static Value applyTransforms(PatternRewriter &b, ThreadwiseWriteAllOp storeOp,
                             Value src) {
  // 0. capture the memref containing the outputs being written
  Location loc = storeOp.getLoc();
  Value gemmOuts = storeOp.getSource();
  auto gemmOutsType = gemmOuts.getType().cast<MemRefType>();

  // 1. create a second allocation of the same type to hold loaded elements
  Value alloc = b.create<GpuAllocOp>(loc, gemmOutsType);

  // 2. clone twcopy for <addend> into regs
  LLVM_DEBUG(llvm::dbgs() << "Src type: " << src.getType() << " dest type: "
                          << storeOp.getDest().getType() << "\n");

  // 2.0. first move all transforms before the relocated linalg.generic
  moveTransformsBefore(b, src);

  // 2.1. apply transform chain from output
  src = transformAccordingly(b, src, storeOp.getDest());

  // 2.2. load into registers
  b.create<ThreadwiseReadIntoOp>(loc, src, alloc, storeOp.getExtraViews(),
                                 storeOp.getForceUnroll(),
                                 storeOp.getUseIndexDiffs());
  return alloc;
}

static Operation *traceToNonViewOp(Operation *op) {
  if (auto transform = dyn_cast<TransformOp>(op)) {
    Value result = transform.getResult();
    // TODO(sjw): fix when divergence is encountered
    assert(result.hasOneUse());
    for (auto &use : result.getUses()) {
      return traceToNonViewOp(use.getOwner());
    }
  }
  return op;
}

static ThreadwiseWriteAllOp traceToThreadwiseWrite(Value inp) {
  // 1. Validate that the only uses of the linalg.generic input are the one
  // generic and a copy operation or transform.
  ThreadwiseWriteAllOp result;
  for (Operation *use : inp.getUsers()) {
    use = traceToNonViewOp(use);
    if (isa<memref::DeallocOp>(use)) {
      // ignore
      continue;
    }
    if (auto lgop = dyn_cast<linalg::GenericOp>(use)) {
      // reader
      if (!llvm::is_contained(lgop.inputs(), inp)) {
        return ThreadwiseWriteAllOp();
      }
    } else if (auto memcpy = dyn_cast<memref::CopyOp>(use)) {
      // reader
      if (memcpy.getOperand(0) != inp) {
        return ThreadwiseWriteAllOp();
      }
    } else if (auto store = dyn_cast<ThreadwiseWriteAllOp>(use)) {
      // Threadwise copy that is already unttransformed (new style)
      if (result) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Multiple global stores somehow, no fusion\n");
        return ThreadwiseWriteAllOp();
      }
      result = store;
    } else {
      return ThreadwiseWriteAllOp();
    }
  }

  if (!result)
    LLVM_DEBUG(llvm::dbgs() << "Align tiling: generic not tracing to copy\n");
  return result;
}

// Returns the value of the buffer that's meant to be the new writeback.
static Value reconfigureLAGeneric(PatternRewriter &b,
                                  linalg::GenericOp laGeneric, Value laIn,
                                  ThreadwiseWriteAllOp twWriteOp) {
  MLIRContext *ctx = laGeneric.getContext();
  Location loc = laGeneric.getLoc();
  auto regType = laIn.getType().template cast<MemRefType>();
  auto laOut = b.create<GpuAllocOp>(loc, regType);

  SmallVector<AffineMap, 5> laGenericAMaps;
  SmallVector<Value, 5> newInputs;
  for (auto inp : laGeneric.inputs()) {
    Value newInput;
    if (traceToThreadwiseWrite(inp)) {
      newInput = laIn;
    } else {
      // 2.1.1. Align tiling of other inputs
      newInput = applyTransforms(b, twWriteOp, inp);
    }
    newInputs.push_back(newInput);
    laGenericAMaps.push_back(AffineMap::getMultiDimIdentityMap(
        newInput.getType().template cast<MemRefType>().getRank(), ctx));
  }

  laGenericAMaps.push_back(
      AffineMap::getMultiDimIdentityMap(regType.getRank(), ctx));

  laGeneric.inputsMutable().assign(newInputs);
  laGeneric.outputsMutable().assign(laOut);

  // 2.2. Reset affine maps
  laGeneric.indexing_mapsAttr(b.getAffineMapArrayAttr(laGenericAMaps));

  // 2.3. Reset iterator types
  SmallVector<StringAttr, 5> laGenericIteratorArr(regType.getRank(),
                                                  b.getStringAttr("parallel"));
  laGeneric.iterator_typesAttr(b.getArrayAttr(ArrayRef<Attribute>(
      laGenericIteratorArr.begin(), laGenericIteratorArr.end())));
  return laOut;
}

static Value findThreadwiseWrite(linalg::GenericOp laGeneric,
                                 ThreadwiseWriteAllOp &twWriteOp) {
  for (auto input : laGeneric.inputs()) {
    if (auto allocOp = input.getDefiningOp<memref::AllocOp>()) {
      if (auto twop = traceToThreadwiseWrite(input)) {
        twWriteOp = twop;
        return input;
      }
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
  for (StringRef iterType :
       laGeneric.iterator_types().getAsValueRange<StringAttr>())
    if (iterType != "parallel")
      return laGeneric.emitError("must be fully parallel");

  Value out = *laGeneric.outputs().begin(); // may be another arg
  // 0.1. Test compatibility,  Only 1 output supported
  if (laGeneric.outputs().size() > 1)
    return laGeneric.emitError("only 1 output supported");

  // 0.2. Sanity check, skip already fused.
  for (auto inp : laGeneric.inputs()) {
    if (auto fusedAlloc = inp.getDefiningOp<GpuAllocOp>()) {
      LLVM_DEBUG(llvm::dbgs() << "Found existing fusion, bailing\n");
      return failure();
    }
  }

  // 1. Trace input to global_store.
  // 1.1. Find the (implicit) gemm output
  ThreadwiseWriteAllOp gemmStoreOp;
  Value laGenericInputLeadingToGemmStore =
      findThreadwiseWrite(laGeneric, gemmStoreOp);
  if (!laGenericInputLeadingToGemmStore)
    return failure();

  auto actualLAGenericOut = laGeneric.getOutputOperand(0);
  if (laGenericInputLeadingToGemmStore.getType() !=
      actualLAGenericOut->get().getType()) {
    // TODO(sjw): fix for mixed types
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

  // 2.2. Tile linalg.generic with vgpr as input, return output vgprs
  Value laOutRegs = reconfigureLAGeneric(b, laGeneric, fusionRegs, gemmStoreOp);
  // 2.2.0. Move the generic before the write-back. This'll put all
  // the copy loops for other inputs before the generic due to insertion
  // order.
  laGeneric->moveBefore(gemmStoreOp);

  gemmOut.replaceAllUsesWith(fusionRegs);

  // 2.3. Replace rock.threadwise_write_all inputs with la.generic result vgprs
  gemmStoreOp.getSourceMutable().assign(laOutRegs);

  out = transformAccordingly(b, out, gemmStoreOp.getDest());
  gemmStoreOp.getDestMutable().assign(out);

  if (auto outAlloc = out.getDefiningOp<memref::AllocOp>()) {
    outAlloc->moveBefore(gemmStoreOp);
  }

  return success();
}

LogicalResult MemcpyRewritePattern::matchAndRewrite(memref::CopyOp copy,
                                                    PatternRewriter &b) const {
  auto src = copy.getSource();
  auto trg = copy.getTarget();

  Operation *gemmStoreOp = nullptr;
  if (auto allocOp = src.getDefiningOp<memref::AllocOp>()) {
    if (auto twop = traceToThreadwiseWrite(src)) {
      gemmStoreOp = twop;
    }
  }

  if (gemmStoreOp && isa<ThreadwiseWriteAllOp>(gemmStoreOp)) {
    PatternRewriter::InsertionGuard guard(b);
    b.setInsertionPoint(gemmStoreOp);

    // 1. replace memref.copy with rock.threadwise_write_all
    trg = transformAccordingly(b, trg, gemmStoreOp->getOperand(1));
    gemmStoreOp->setOperand(1, trg);

    if (auto outAlloc = trg.getDefiningOp<memref::AllocOp>())
      outAlloc->moveBefore(gemmStoreOp);

    b.eraseOp(copy);
    return success();
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

void RockLinalgAlignPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  {
    RewritePatternSet patterns(ctx);
    patterns.add<LAGenericRewritePattern>(ctx);
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
