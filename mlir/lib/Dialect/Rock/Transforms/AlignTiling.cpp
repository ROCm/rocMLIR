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
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
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

static void insertLoadFromOtherSource(PatternRewriter &b, Location loc,
                                      ThreadwiseWriteAllOp gemmStoreOp,
                                      Value src, Value dest) {
  LLVM_DEBUG(llvm::dbgs() << "Src type: " << src.getType() << " dest type: "
                          << gemmStoreOp.getDest().getType() << "\n");

  Operation *cop = b.create<ThreadwiseReadIntoOp>(
      loc, src, dest, gemmStoreOp.getExtraViews(), gemmStoreOp.getForceUnroll(),
      gemmStoreOp.getUseIndexDiffs());

  while (auto sop = src.getDefiningOp()) {
    if (auto rtop = dyn_cast<rock::TransformOp>(sop)) {
      sop->moveBefore(cop);
      cop = sop;
      src = rtop.getOperand();
    } else {
      assert(0);
    }
  }
}

static Value makeTransformingCopyLoop(PatternRewriter &b,
                                      ThreadwiseWriteAllOp storeOp, Value inp) {
  // 0. capture the memref containing the outputs being written
  Location loc = storeOp.getLoc();
  Value gemmOuts = storeOp.getOperand(0);
  auto gemmOutsType = gemmOuts.getType().cast<MemRefType>();

  // 1. create a second allocation of the same type to hold loaded elements
  Value alloc = b.create<GpuAllocOp>(loc, gemmOutsType);

  // 2. clone twcopy for <addend> -> regs as transforming_for
  insertLoadFromOtherSource(b, loc, storeOp, inp, alloc);
  return alloc;
}

Value applyTransforms(PatternRewriter &b, ThreadwiseWriteAllOp gemmStoreOp,
                      Value inp, AffineMap outToInpMap) {
  Value ret = inp;

  // 1. insert broadcast op if necessary
  // MemRefType outType = gemmStoreOp.getDest().getType();
  // assert(outType == inp.getType());
  // ret = insertTransposeAndBroadcastTransforms(b, outType.getShape(), ret,
  //                                             outToInpMap);

  // 2. also create global_store from global to regs
  return makeTransformingCopyLoop(b, gemmStoreOp, ret);
}

static Operation *traceToRealOp(Operation *op) {
  if (auto transform = dyn_cast<rock::TransformOp>(op)) {
    Value result = transform.getResult();
    if (result.hasOneUse()) {
      for (auto &use : result.getUses()) {
        return traceToRealOp(use.getOwner());
      }
    }
  }
  return op;
}

static rock::ThreadwiseWriteAllOp traceToGlobalStore(Value inp) {
  // 1. Validate that the only uses of the linalg.generic input are the one
  // generic and a copy operation or transform.
  bool allValidUses = true;
  rock::ThreadwiseWriteAllOp result;
  for (Operation *use : inp.getUsers()) {
    use = traceToRealOp(use);
    if (isa<memref::DeallocOp>(use)) {
      // ignore
      continue;
    }
    if (isa<linalg::GenericOp>(use)) {
      // reader
    } else if (auto memcpy = dyn_cast<memref::CopyOp>(use)) {
      // reader
      if (memcpy.getOperand(0) != inp) {
        allValidUses = false;
      }
    } else if (auto store = dyn_cast<rock::ThreadwiseWriteAllOp>(use)) {
      // Threadwise copy that is already unttransformed (new style)
      if (result) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Multiple global stores somehow, no fusion\n");
        allValidUses = false;
      }
      result = store;
    } else {
      allValidUses = false;
    }
  }

  if (!result)
    LLVM_DEBUG(llvm::dbgs() << "Align tiling: generic not tracing to copy\n");
  if (!allValidUses)
    LLVM_DEBUG(llvm::dbgs() << "Align tiling: found invalid use\n");
  return allValidUses ? result : rock::ThreadwiseWriteAllOp();
}

// Returns the value of the buffer that's meant to be the new writeback.
static Value reconfigureLAGeneric(PatternRewriter &b,
                                  linalg::GenericOp laGeneric, Value laIn,
                                  ArrayRef<AffineMap> idxMaps,
                                  rock::ThreadwiseWriteAllOp gemmGlobalStore) {
  MLIRContext *ctx = laGeneric.getContext();
  Location loc = laGeneric.getLoc();
  auto regType = laIn.getType().template cast<MemRefType>();
  auto laOut = b.create<GpuAllocOp>(loc, regType);

  AffineMap outIdxMap = idxMaps.back();
  auto invertOutIdxMap = inversePermutation(outIdxMap);
  SmallVector<AffineMap, 5> laGenericAMaps;
  SmallVector<Value, 5> newInputs;
  for (auto pair : llvm::zip(laGeneric.inputs(), idxMaps)) {
    if (Value inp = std::get<0>(pair)) {
      AffineMap inpIdxMap = std::get<1>(pair);
      auto outToInMap = inpIdxMap.compose(invertOutIdxMap);
      Value newInput;
      if (traceToGlobalStore(inp)) {
        newInput = laIn;
      } else {
        // 2.1.1. Align tiling of other inputs
        newInput = applyTransforms(b, gemmGlobalStore, inp, outToInMap);
      }
      newInputs.push_back(newInput);
      laGenericAMaps.push_back(AffineMap::getMultiDimIdentityMap(
          newInput.getType().template cast<MemRefType>().getRank(), ctx));
    }
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

static Value findGlobalStore(linalg::GenericOp laGeneric,
                             rock::ThreadwiseWriteAllOp &gemmStoreOp) {
  for (auto input : laGeneric.inputs()) {
    if (auto allocOp = input.getDefiningOp<memref::AllocOp>()) {
      if (auto twop = traceToGlobalStore(input)) {
        gemmStoreOp = twop;
        return input;
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "No input is leading to a global store.\n");
  return Value();
}

static Value transformGenericOutput(PatternRewriter &b, Value nOut,
                                    Value gemmOut) {
  if (auto rtop = dyn_cast<rock::TransformOp>(gemmOut.getDefiningOp())) {
    nOut = transformGenericOutput(b, nOut, rtop.getOperand());

    // clone
    BlockAndValueMapping cmap;
    cmap.map(rtop.getOperand(), nOut);
    auto ntop = dyn_cast<rock::TransformOp>(b.clone(*rtop, cmap));
    nOut = ntop.getResult();
  }

  return nOut;
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
      return failure();

  Value out = *laGeneric.outputs().begin(); // may be another arg
  // 0.1. Test compatibility,  Only 1 output supported
  if (laGeneric.outputs().size() > 1)
    return failure();

  // 0.2. Sanity check, skip already fused.
  for (auto inp : laGeneric.inputs()) {
    if (auto fusedAlloc = inp.getDefiningOp<GpuAllocOp>()) {
      LLVM_DEBUG(llvm::dbgs() << "Found existing fusion, bailing\n");
      return failure();
    }
  }

  if (laGeneric.getNumOutputs() != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "Currently, multi-output fusion is not supported.\n");
    return failure();
  }

  // 1. Trace input to global_store.
  // 1.1. Find the (implicit) gemm output
  rock::ThreadwiseWriteAllOp gemmStoreOp;
  Value laGenericInputLeadingToGlobalStore =
      findGlobalStore(laGeneric, gemmStoreOp);
  if (!laGenericInputLeadingToGlobalStore)
    return failure();

  auto actualLAGenericOut = laGeneric.getOutputOperand(0);
  auto actualLAGenericOutIdxMap =
      laGeneric.getTiedIndexingMap(actualLAGenericOut);
  auto invertOutIdxMap = inversePermutation(actualLAGenericOutIdxMap);
  if (laGenericInputLeadingToGlobalStore.getType() !=
      actualLAGenericOut->get().getType()) {
    LLVM_DEBUG(llvm::dbgs() << "Currently, we assume the shape of gemmStore op "
                               "and linalg output is the same.\n");
    LLVM_DEBUG(llvm::dbgs()
               << "This instance it differs : "
               << laGenericInputLeadingToGlobalStore.getType() << " vs "
               << actualLAGenericOut->get().getType() << " .\n");
    return failure();
  }

  SmallVector<AffineMap> idxMaps = laGeneric.getIndexingMapsArray();
  for (auto pair : llvm::zip(idxMaps, laGeneric.inputs())) {
    AffineMap inpIdxMap = std::get<0>(pair);
    auto outToInMap = inpIdxMap.compose(invertOutIdxMap);
    Value inp = std::get<1>(pair);
    if (inp != laGenericInputLeadingToGlobalStore) {
      SmallVector<unsigned> permutedDims;
      if (!outToInMap.isProjectedPermutation(/*allowZeroInResults=*/true)) {
        LLVM_DEBUG(llvm::dbgs() << outToInMap << "\n");
        LLVM_DEBUG(llvm::dbgs() << "^ is not a isProjectedPermutation from "
                                   "output coords to fusion input\n");
        return failure();
      }
    }
  }
  if (!gemmStoreOp) {
    LLVM_DEBUG(llvm::dbgs() << "Align tiling: couldn't find writeback\n");
    return failure();
  }
  // 2. Apply if input found
  Value gemmOut = gemmStoreOp.getOperand(0);
  auto gemmOutType = gemmOut.getType().cast<MemRefType>();

  PatternRewriter::InsertionGuard guard(b);
  // 2.0. Reset insertion point to before the copy.
  b.setInsertionPoint(gemmStoreOp);

  // 2.1. Take out a slice of the result vector to create a vector-sized
  // slice to enable creating the fusion section.
  Value fusionRegs = b.create<GpuAllocOp>(loc, gemmOutType);

  // 2.2. Tile linalg.generic with vgpr as input, return output vgprs
  Value laOutRegs =
    reconfigureLAGeneric(b, laGeneric, fusionRegs, idxMaps, gemmStoreOp);
  // 2.2.0. Move the generic before the write-back. This'll put all
  // the copy loops for other inputs before the generic due to insertion
  // order.
  laGeneric->moveBefore(gemmStoreOp);

  // 2.3. Replace twcopy inputs with la.generic result vgprs

  gemmStoreOp.setOperand(0, laOutRegs);

  out = transformGenericOutput(b, out, gemmStoreOp.getOperand(1));
  gemmStoreOp.setOperand(1, out);

  if (auto outAlloc = out.getDefiningOp<memref::AllocOp>()) {
    outAlloc->moveBefore(gemmStoreOp);
  }

  return success();
}

LogicalResult MemcpyRewritePattern::matchAndRewrite(memref::CopyOp copy,
                                                    PatternRewriter &b) const {
  Location loc = copy.getLoc();

  auto src = copy.getSource();
  auto trg = copy.getTarget();

  Operation *gemmStoreOp = nullptr;
  if (auto allocOp = src.getDefiningOp<memref::AllocOp>()) {
    if (auto twop = traceToGlobalStore(src)) {
      gemmStoreOp = twop;
    }
  }

  if (gemmStoreOp && isa<rock::ThreadwiseWriteAllOp>(gemmStoreOp)) {
    PatternRewriter::InsertionGuard guard(b);
    // 2.0. Reset insertion point to before the copy.
    b.setInsertionPoint(gemmStoreOp);

    trg = transformGenericOutput(b, trg, gemmStoreOp->getOperand(1));
    gemmStoreOp->setOperand(1, trg);

    if (auto outAlloc = trg.getDefiningOp<memref::AllocOp>())
      outAlloc->moveBefore(gemmStoreOp);

    b.eraseOp(copy);
    return success();
  }

  return failure();
}

static bool isUnfusedKernelStore(rock::ThreadwiseWriteAllOp store) {
  bool ret = isa_and_nonnull<memref::AllocOp>(store.getDest().getDefiningOp());
  if (ret) {
    store.getDest().getDefiningOp()->emitOpError(
        "could not use fusion to eliminate this intermediate buffer. Kernel "
        "compilation canot proceed");
  }
  return ret;
}

/// FIXME: This rewrite should be after fusion. However, since the fusion
/// refactor hasn't landed yet and since we need to preserve the structure of
/// the existing fusion code, put the threadwise_write_all expander here.
namespace {
struct ThreadwiseReadIntoRewritePattern
    : public OpConversionPattern<ThreadwiseReadIntoOp> {
  using OpConversionPattern<ThreadwiseReadIntoOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ThreadwiseReadIntoOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final;
};

struct ThreadwiseWriteAllRewritePattern
    : public OpConversionPattern<ThreadwiseWriteAllOp> {
  using OpConversionPattern<ThreadwiseWriteAllOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ThreadwiseWriteAllOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final;
};
} // end anonymous namespace

LogicalResult ThreadwiseReadIntoRewritePattern::matchAndRewrite(
    ThreadwiseReadIntoOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &b) const {
  Location loc = op.getLoc();
  TypedValue<MemRefType> sourceView = adaptor.getSource();
  TypedValue<MemRefType> dest = adaptor.getDest();

  auto [buffer, transforms] = untransform(b, sourceView, op.getExtraViews());

  int64_t numValues = dest.getType().getNumElements();
  ArrayRef<int64_t> bufferShape =
      buffer.getType().cast<ShapedType>().getShape();

  // We are vectorizing in the iter dimension, not block ID or thread ID
  int64_t vectorLen =
      getMaxVectorization(transforms, /*dim=*/2, numValues, bufferShape);
  LLVM_DEBUG(llvm::dbgs() << "Max vectorization for read_into = " << vectorLen
                          << "\n");
  auto [leftOobDims, rightOobDims] = computeOobFromTransforms(b, transforms);

  Type loadType =
      vectorTypeOrSelf(sourceView.getType().getElementType(), vectorLen);
  bool forceUnroll = op.getForceUnroll();
  bool useIndexDiffs = op.getUseIndexDiffs();

  // Constant / consistent arguments
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  Value bid = b.createOrFold<rock::WorkgroupIdOp>(loc, b.getIndexType());
  Value tid = b.createOrFold<rock::WorkitemIdOp>(loc, b.getIndexType());

  SmallVector<Value, 3> readStartCoords = {bid, tid, zero};

  auto loadLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{readStartCoords, readStartCoords},
      ArrayRef<Attribute>{transforms, b.getArrayAttr({})},
      ArrayRef<int64_t>{1, 1, numValues}, ArrayRef<int64_t>{1, 1, vectorLen},
      forceUnroll, useIndexDiffs);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loadLoop.getBody());
    Value loaded =
        b.create<GlobalLoadOp>(loc, loadType, buffer, leftOobDims, rightOobDims,
                               loadLoop.getLowerCoords(/*domain=*/0));
    b.create<InBoundsStoreOp>(loc, loaded, dest,
                              loadLoop.getLowerCoords(/*domain=*/1)[2]);
  }
  b.eraseOp(op);
  return success();
}

LogicalResult ThreadwiseWriteAllRewritePattern::matchAndRewrite(
    ThreadwiseWriteAllOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &b) const {
  Location loc = op.getLoc();
  TypedValue<MemRefType> source = adaptor.getSource();
  TypedValue<MemRefType> destView = adaptor.getDest();

  auto [buffer, transforms] = untransform(b, destView, op.getExtraViews());

  int64_t numValues = source.getType().getNumElements();
  ArrayRef<int64_t> bufferShape =
      buffer.getType().cast<ShapedType>().getShape();

  // We are vectorizing in the iter dimension, not block ID or thread ID
  int64_t vectorLen =
      getMaxVectorization(transforms, /*dim=*/2, numValues, bufferShape);
  LLVM_DEBUG(llvm::dbgs() << "Max vectorization for write_all = " << vectorLen
                          << "\n");
  auto [leftOobDims, rightOobDims] = computeOobFromTransforms(b, transforms);

  bool forceUnroll = op.getForceUnroll();
  bool useIndexDiffs = op.getUseIndexDiffs();

  // Constant / consistent arguments
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  Value bid = b.createOrFold<rock::WorkgroupIdOp>(loc, b.getIndexType());
  Value tid = b.createOrFold<rock::WorkitemIdOp>(loc, b.getIndexType());

  SmallVector<Value, 3> writeStartCoords = {bid, tid, zero};

  auto outLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{writeStartCoords, writeStartCoords},
      ArrayRef<Attribute>{b.getArrayAttr({}), transforms},
      ArrayRef<int64_t>{1, 1, numValues}, ArrayRef<int64_t>{1, 1, vectorLen},
      forceUnroll, useIndexDiffs);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(outLoop.getBody());
    b.create<GlobalStoreOp>(loc, source, buffer, b.getIndexAttr(vectorLen),
                            op.getStoreMethodAttr(), leftOobDims, rightOobDims,
                            outLoop.getLowerCoords(/*domain=*/0)[2],
                            outLoop.getLowerCoords(/*domain=*/1));
  }
  b.eraseOp(op);
  return success();
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
        getOperation().walk([](rock::ThreadwiseWriteAllOp store) {
          return isUnfusedKernelStore(store) ? WalkResult::interrupt()
                                             : WalkResult::advance();
        });
    if (verifyAllStores.wasInterrupted())
      signalPassFailure();
  }

  {
    ConversionTarget writeAllTarget(*ctx);
    writeAllTarget.addIllegalOp<ThreadwiseReadIntoOp, ThreadwiseWriteAllOp>();
    writeAllTarget.addLegalDialect<arith::ArithmeticDialect, rock::RockDialect>();
    RewritePatternSet writeAllPatterns(ctx);
    writeAllPatterns
      .add<ThreadwiseReadIntoRewritePattern, ThreadwiseWriteAllRewritePattern>(
                                                                               ctx);
    if (failed(applyPartialConversion(getOperation(), writeAllTarget,
                                      std::move(writeAllPatterns))))
      signalPassFailure();

    OpPassManager cleanupPasses("func.func");
    cleanupPasses.addPass(mlir::createCanonicalizerPass());
    (void)runPipeline(cleanupPasses, getOperation());
  }
}
