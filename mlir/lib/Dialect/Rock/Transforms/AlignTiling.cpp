//===- AlignTiling.cpp - Align Linalg ops with Rock ops
//------------------===//
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
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

struct MILARewritePattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp laGeneric,
                                PatternRewriter &b) const override;
};
} // end anonymous namespace

static void insertLoadFromOtherSource(PatternRewriter &b, Location loc,
                                      GlobalStoreOp gemmStoreOp, Value srcOp,
                                      Value dest) {
  LLVM_DEBUG(llvm::dbgs() << "Src type: " << srcOp.getType() << " dest type: "
                          << gemmStoreOp.getDest().getType() << "\n");
  SmallVector<Value, 6> loadCoord = gemmStoreOp.getDestCoord();
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);

  auto writeCLoop = gemmStoreOp->getParentOfType<TransformingForOp>();
  assert(writeCLoop && "global_store must be in a transforming_for");

  // Handle broadcasts introduced during fusion.
  ArrayAttr sourceTransformsFromOp;
  Value source;
  std::tie(source, sourceTransformsFromOp) = untransform(b, srcOp);

  int64_t copyLength = gemmStoreOp.getLength().getSExtValue();
  Type destElemType = dest.getType().cast<MemRefType>().getElementType();

  // In general, note that keeping the vectorization of the writeback is safe
  // on account of the fact that vectorization means that the maps for the
  // gemm output (and thus the extra argument) are contiguous in the
  // underlying memory.

  // If there are no broadcasts, re-use the coordianes for the writeback
  if (sourceTransformsFromOp.empty()) {
    Type typeToLoad = destElemType;
    if (copyLength > 1)
      typeToLoad = VectorType::get({copyLength}, typeToLoad);

    Value loaded = b.create<GlobalLoadOp>(loc, typeToLoad, source, loadCoord);
    b.create<InBoundsStoreOp>(loc, loaded, dest, zero);
  } else {
    // Note: the vectorization of extra argument may be smaller than the
    // vectorization of the convolution.
    size_t extraMapInSize = loadCoord.size();

    int64_t lastDim = extraMapInSize - 1;
    int64_t maxVectorLen =
        getMaxVectorization(sourceTransformsFromOp, lastDim, copyLength,
                            source.getType().cast<MemRefType>().getShape());

    SmallVector<int64_t> bounds(extraMapInSize, 1LL),
        strides(extraMapInSize, 1LL);
    bounds[lastDim] = copyLength;
    strides[lastDim] = maxVectorLen;

    SmallVector<Value> zeroes(extraMapInSize, zero);

    Type typeToLoad = destElemType;
    if (maxVectorLen > 1)
      typeToLoad = VectorType::get(maxVectorLen, typeToLoad);

    auto copyLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{loadCoord, zeroes},
        ArrayRef<Attribute>{sourceTransformsFromOp, b.getArrayAttr({})},
        /*bounds=*/ArrayRef<int64_t>(bounds),
        /*strides=*/ArrayRef<int64_t>(strides), /*forceUnroll=*/true,
        /*useIndexDiffs=*/true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(copyLoop.getBody());
      Value loaded = b.create<GlobalLoadOp>(
          loc, typeToLoad, source, copyLoop.getLowerCoords(/*domain=*/0));
      b.create<InBoundsStoreOp>(loc, loaded, dest,
                                copyLoop.getLowerCoords(/*domain=*/1)[lastDim]);
    }
  }
}

static Value makeTransformingCopyLoop(PatternRewriter &b, GlobalStoreOp storeOp,
                                      Value inp) {
  // 0. capture the memref containing the outputs being written
  Location loc = storeOp.getLoc();
  Value gemmOuts = storeOp.getSource();
  auto gemmOutsType = gemmOuts.getType().cast<MemRefType>();
  int64_t sliceLength = storeOp.getLength().getSExtValue();
  auto sliceLengthType = gemmOutsType.clone(sliceLength).cast<MemRefType>();

  // 1. create a second allocation of the same type to hold loaded elements
  Value alloc = b.create<GpuAllocOp>(loc, sliceLengthType);

  // 2. clone twcopy for <addend> -> regs as transforming_for
  insertLoadFromOtherSource(b, loc, storeOp, inp, alloc);
  return alloc;
}

Value applyTransforms(PatternRewriter &b, GlobalStoreOp gemmStoreOp, Value inp,
                      AffineMap outToInpMap) {
  Value ret = inp;

  // 1. insert broadcast op if necessary
  MemRefType outType = gemmStoreOp.getDest().getType();
  ret = insertTransposeAndBroadcastTransforms(b, outType.getShape(), ret,
                                              outToInpMap);

  // 2. also create global_store from global to regs
  //    TODO(sjw): make sure output buffer writes (means these inputs will be
  //    buffer reads)
  return makeTransformingCopyLoop(b, gemmStoreOp, ret);
}

static GlobalStoreOp traceToGlobalStore(Value inp) {
  // 1. Validate that the only uses of the linalg.generic input are the one
  // generic and a copy operation or transform.
  bool allValidUses = true;
  GlobalStoreOp result;
  for (Operation *use : inp.getUsers()) {
    if (isa<memref::DeallocOp>(use)) {
      // ignore
      continue;
    }
    if (isa<linalg::GenericOp>(use)) {
      // reader
    } else if (auto store = dyn_cast<GlobalStoreOp>(use)) {
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
  return allValidUses ? result : GlobalStoreOp();
}

// Returns the value of the buffer that's meant to be the new writeback.
static Value reconfigureLAGeneric(PatternRewriter &b,
                                  linalg::GenericOp laGeneric, Value laIn,
                                  ArrayRef<AffineMap> idxMaps,
                                  GlobalStoreOp gemmGlobalStore) {
  MLIRContext *ctx = laGeneric.getContext();
  Location loc = laGeneric.getLoc();
  Value twout = gemmGlobalStore.getDest();
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
      if (inp == twout) {
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

static LogicalResult findGlobalStore(linalg::GenericOp laGeneric,
                                     Value &inputLeadingToGlobalStore,
                                     GlobalStoreOp &gemmStoreOp) {
  for (auto pair :
       llvm::zip(laGeneric.inputs(), laGeneric.getIndexingMapsArray())) {
    AffineMap inpIdxMap = std::get<1>(pair);
    Value input = std::get<0>(pair);
    GlobalStoreOp maybeStore = traceToGlobalStore(input);
    if (maybeStore) {
      if (gemmStoreOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Multiple generic inputs come from writeback\n");
        return failure();
      }

      auto laGenericOut = laGeneric.getOutputOperand(0);
      auto laGenericOutIdxMap = laGeneric.getTiedIndexingMap(laGenericOut);
      auto invertOutIdxMap = inversePermutation(laGenericOutIdxMap);
      auto outToInMap = inpIdxMap.compose(invertOutIdxMap);
      SmallVector<unsigned> permutedDims;
      // This is not to allow broadcasting but due to canonical linalg
      // form if the unit dims can carry affine expr to be zero in the
      // translation, hence there is a following check.
      if (!outToInMap.isMinorIdentityWithBroadcasting(&permutedDims)) {
        LLVM_DEBUG(llvm::dbgs() << outToInMap << "\n");
        LLVM_DEBUG(
            llvm::dbgs()
            << "The store is not even a minor identity with broadcasting.\n");
        return failure();
      }
      auto inpShape = input.getType().cast<ShapedType>().getShape();
      LLVM_DEBUG(llvm::dbgs() << "outToInMap = " << outToInMap << "\n");
      LLVM_DEBUG(llvm::dbgs() << "inp shape = "
                              << input.getType().cast<ShapedType>() << "\n");
      LLVM_DEBUG(llvm::dbgs() << "permutedDims = ");
      LLVM_DEBUG(llvm::interleaveComma(permutedDims, llvm::dbgs()));
      LLVM_DEBUG(llvm::dbgs() << "\n");

      for (auto bDim : permutedDims) {
        if (inpShape[bDim] != 1) {
          LLVM_DEBUG(llvm::dbgs()
                     << "The store input cannot be a real broacast.\n");
          return failure();
        };
      }
      gemmStoreOp = maybeStore;
      inputLeadingToGlobalStore = input;
    }
  }
  if (!gemmStoreOp) {
    LLVM_DEBUG(llvm::dbgs() << "No input is leading to a global store.\n");
    return failure();
  }
  return success();
}

LogicalResult MILARewritePattern::matchAndRewrite(linalg::GenericOp laGeneric,
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
  GlobalStoreOp gemmStoreOp;
  Value laGenericInputLeadingToGlobalStore;
  if (failed(findGlobalStore(laGeneric, laGenericInputLeadingToGlobalStore,
                             gemmStoreOp))) {
    return failure();
  }
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

  Value gemmOuts = gemmStoreOp.getSource();
  auto gemmOutsType = gemmOuts.getType().cast<MemRefType>();
  {
    PatternRewriter::InsertionGuard guard(b);
    // 2.0. Reset insertion point to before the copy.
    b.setInsertionPoint(gemmStoreOp);
    Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);

    // 2.1. Take out a slice of the result vector to create a vector-sized
    // slice to enable creating the fusion section.
    int64_t sliceLength = gemmStoreOp.getLength().getSExtValue();
    MemRefType sliceType = gemmOutsType.clone(sliceLength).cast<MemRefType>();
    Value fusionSlice = b.create<GpuAllocOp>(loc, sliceType);
    Type typeToCopy = sliceType.getElementType();
    if (sliceType.getNumElements() > 1)
      typeToCopy =
          VectorType::get(sliceType.getShape(), sliceType.getElementType());
    Value sliceVals = b.create<InBoundsLoadOp>(loc, typeToCopy, gemmOuts,
                                               gemmStoreOp.getSourceCoord());
    b.create<InBoundsStoreOp>(loc, sliceVals, fusionSlice, zero);

    // 2.2. Tile linalg.generic with vgpr as input, return output vgprs
    Value laOutRegs =
        reconfigureLAGeneric(b, laGeneric, fusionSlice, idxMaps, gemmStoreOp);
    // 2.2.0. Move the generic before the write-back. This'll put all
    // the copy loops for other inputs before the generic due to insertion
    // order.
    laGeneric->moveBefore(gemmStoreOp);

    // 2.3. Replace twcopy inputs with la.generic result vgprs

    // Since the threadwise copy arg has gone through untransform()
    // its expected output type is the same as the output type of the
    // linalg.generic.
    gemmStoreOp.getSourceMutable().assign(laOutRegs);
    // The indexing has been moved into slice creation, reset source
    // coord.
    gemmStoreOp.getSourceCoordMutable().assign(zero);
    gemmStoreOp.getDestMutable().assign(out);

    return success();
  }

  return failure();
}

static bool isUnfusedKernelStore(rock::GlobalStoreOp store) {
  bool ret = isa_and_nonnull<memref::AllocOp>(store.getDest().getDefiningOp());
  if (ret) {
    store.getDest().getDefiningOp()->emitOpError(
        "could not use fusion to eliminate this intermediate buffer. Kernel "
        "compilation canot proceed");
  }
  return ret;
}

void RockLinalgAlignPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<MILARewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
  WalkResult verifyAllStores =
      getOperation().walk([](rock::GlobalStoreOp store) {
        return isUnfusedKernelStore(store) ? WalkResult::interrupt()
                                           : WalkResult::advance();
      });
  if (verifyAllStores.wasInterrupted())
    signalPassFailure();
}
