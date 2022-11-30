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

// This rewrite will rewrite the linalg IO that has view like-ops surrounding
// them to be consumed by the linalg operation itself adjusting the indexing
// maps to faithfully represent them.
struct InlineViewLikeOperandsLinalgRewritePattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp laGeneric,
                                PatternRewriter &rewriter) const override;
};

// This function will take strides calculated using reassociation of dimensions
// that are modified by view-like ops and recompute the affine expr from src
// view to dest view.
static void getDelinearizedAffineExpr(mlir::ArrayRef<int64_t> strides,
                                      mlir::ArrayRef<int64_t> shapes,
                                      Builder &b, unsigned int position,
                                      SmallVectorImpl<mlir::AffineExpr> &res) {
  AffineExpr resultExpr = b.getAffineDimExpr(position);
  int64_t rank = strides.size();
  // If the rank is 1, expand or collapse shapes will just
  // pass-through the dimensions.
  if (rank == 1) {
    res[0] = resultExpr;
    return;
  }
  for (int i = 0; i < rank; i++) {
    // If the current shape is 1 and the rank is non-zero,
    // could only mean it is being broadcasted. Hence,
    // putting zero.
    if (shapes[i] == 1) {
      res[i] = resultExpr * 0;
    } else {
      // Recording the vector offsets here.
      res[i] = resultExpr;
      // There is no point of putting a modulo if the size
      // is equivalent to that.
      if (i - 1 >= 0 && shapes[i] != strides[i - 1]) {
        res[i] = res[i] % strides[i - 1];
      }

      if (shapes[i] > strides[i]) {
        // We only need the floorDiv if the dimSize
        // is larger than the stride
        res[i] = res[i].floorDiv(strides[i]);
      } else if (shapes[i] < strides[i]) {
        // if the shape is smaller than the stride
        // expr might as well be zero.
        res[i] = res[i] * 0;
      }

      // The resultExpr has to propagated anyway for
      // other dimensions where the recording in the above
      // will do the neccesary checks to remove the modulo
      if (i - 1 >= 0) {
        resultExpr = resultExpr % strides[i - 1];
      }
      resultExpr = resultExpr.floorDiv(strides[i]);
    }
  }
  return;
}

// This function will create a affine map that represent the mapping
// from higher rank memref type to lower rank memref type.
static AffineMap createHigherToLowerRankViewAffineMap(
    PatternRewriter &rewriter,
    ArrayRef<ReassociationIndices> reassociationIndices,
    const MemRefType &higherRankType, const MemRefType &lowerRankType) {
  SmallVector<AffineExpr, 4> resultExprs;
  int iDimCount = 0;
  for (const SmallVector<int64_t, 2> &groups : reassociationIndices) {
    assert(!groups.empty() && "association indices groups cannot be empty");
    unsigned groupSize = groups.size();
    SmallVector<int64_t> suffixProduct(groupSize);
    // Calculate suffix product for all collapse op source dimension sizes.
    suffixProduct[groupSize - 1] = 1;
    for (unsigned i = groupSize - 1; i > 0; i--)
      suffixProduct[i - 1] =
          suffixProduct[i] * higherRankType.getDimSize(groups[i]);
    SmallVector<int64_t> shapes(groupSize);
    for (unsigned i = 0; i < groupSize; i++) {
      shapes[i] = higherRankType.getDimSize(groups[i]);
    }
    // Derive the index values along all dimensions of the source
    // corresponding to the index wrt to collapsed shape op output.
    SmallVector<AffineExpr, 4> srcIndexExpr(suffixProduct.size());
    getDelinearizedAffineExpr(suffixProduct, shapes, rewriter, iDimCount++,
                              srcIndexExpr);
    for (unsigned i = 0; i < groupSize; i++) {
      resultExprs.push_back(srcIndexExpr[i]);
    }
  }
  auto representativeMap = AffineMap::get(
      /*numDims=*/lowerRankType.cast<ShapedType>().getShape().size(),
      /*numSymbols=*/0, resultExprs, rewriter.getContext());
  return representativeMap;
}

// This function will traverse the operands of the linalg.generic folding
// the view like ops to the indexing maps and returning the ultimate
// folded map as well as the root operand.
LogicalResult
foldViewLikeOperands(PatternRewriter &rewriter, Value op, AffineMap &foldedMap,
                     Value &rootOp,
                     SmallVectorImpl<Operation *> &toBeErasedViewLikeOps) {
  if (memref::CollapseShapeOp collapseOp =
          op.getDefiningOp<memref::CollapseShapeOp>()) {
    SmallVector<mlir::ReassociationIndices, 4U> reassociationIndices =
        collapseOp.getReassociationIndices();
    MemRefType lowerRankType = collapseOp.getType();
    MemRefType higherRankType = collapseOp.getSrcType();
    auto representativeMap = createHigherToLowerRankViewAffineMap(
        rewriter, reassociationIndices, higherRankType, lowerRankType);
    foldedMap = representativeMap.compose(foldedMap);
    toBeErasedViewLikeOps.push_back(collapseOp);
    return foldViewLikeOperands(rewriter, collapseOp.getViewSource(), foldedMap,
                                rootOp, toBeErasedViewLikeOps);
  }
  if (memref::ExpandShapeOp expandOp =
          op.getDefiningOp<memref::ExpandShapeOp>()) {
    SmallVector<mlir::ReassociationIndices, 4U> reassociationIndices =
        expandOp.getReassociationIndices();
    MemRefType higherRankType = expandOp.getType();
    MemRefType lowerRankType = expandOp.getSrcType();
    auto representativeMap = createHigherToLowerRankViewAffineMap(
        rewriter, reassociationIndices, higherRankType, lowerRankType);
    // We take the inverse here because in expand shape it is going from lower
    // to higher rank.
    representativeMap = inversePermutation(representativeMap);
    foldedMap = representativeMap.compose(foldedMap);
    toBeErasedViewLikeOps.push_back(expandOp);
    return foldViewLikeOperands(rewriter, expandOp.getViewSource(), foldedMap,
                                rootOp, toBeErasedViewLikeOps);
  }
  rootOp = op;
  return success();
}

LogicalResult InlineViewLikeOperandsLinalgRewritePattern::matchAndRewrite(
    linalg::GenericOp laGeneric, PatternRewriter &rewriter) const {
  Location loc = laGeneric.getLoc();
  int ioCount = 0;

  SmallVector<AffineMap, 4U> newMaps;
  SmallVector<Value, 4U> newInputs;
  unsigned int changedIOCount = 0;

  SmallVector<Operation *> toBeErasedViewLikeOps;
  for (auto input : laGeneric.inputs()) {
    AffineMap resMap = laGeneric.getIndexingMapsArray()[ioCount++];
    Value rootOp;
    if (foldViewLikeOperands(rewriter, input, resMap, rootOp,
                             toBeErasedViewLikeOps)
            .failed()) {
      return failure();
    }
    if (rootOp != input)
      changedIOCount++;
    newInputs.push_back(rootOp);
    newMaps.push_back(resMap);
  }
  SmallVector<Value, 4U> newOutputs;
  for (auto output : laGeneric.outputs()) {
    AffineMap resMap = laGeneric.getIndexingMapsArray()[ioCount++];
    Value rootOp;
    if (foldViewLikeOperands(rewriter, output, resMap, rootOp,
                             toBeErasedViewLikeOps)
            .failed()) {
      return failure();
    }
    if (rootOp != output)
      changedIOCount++;
    newOutputs.push_back(rootOp);
    newMaps.push_back(resMap);
  }
  if (changedIOCount == 0)
    return failure();

  SmallVector<StringRef> iteratorTypes = llvm::to_vector<4>(
      laGeneric.getIteratorTypes().getAsValueRange<StringAttr>());
  auto newLaGenericOp = rewriter.create<linalg::GenericOp>(
      loc, newInputs, newOutputs, newMaps, iteratorTypes);
  rewriter.inlineRegionBefore(laGeneric->getRegion(0),
                              newLaGenericOp.getRegion(),
                              newLaGenericOp.getRegion().begin());
  rewriter.replaceOp(laGeneric, newLaGenericOp.getResults());
  for (auto op : toBeErasedViewLikeOps) {
    rewriter.eraseOp(op);
  }
  return success();
}

struct MILARewritePattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp laGeneric,
                                PatternRewriter &b) const override;
};
} // end anonymous namespace

// This function will create a permutation that will permute the originalMap to
// be a MinorIdentityWithBroadcast. This is used to add a permutation later in
// the chain.
// e.g. :
// (d0, d1, d2, d4) -> (0, d1) was the map
// This function will return a permutation : [0, 3, 2, 1] s.t.
// apply it to the original map would result in
// (d0, d4, d2, d1) -> (0, d1) in effect.
static void createPermutationForMinorIdentityWithBroadcast(
    const AffineMap &originalMap, SmallVectorImpl<uint32_t> &perm) {
  for (uint32_t i = 0; i < originalMap.getNumInputs(); ++i) {
    perm.push_back(i);
  }

  llvm::SmallSet<uint32_t, 4> foundInputDims;
  for (const auto &idxAndValue : llvm::enumerate(originalMap.getResults())) {
    auto idx = idxAndValue.index();
    AffineExpr resultExpr = idxAndValue.value();
    if (resultExpr.isa<AffineDimExpr>()) {
      foundInputDims.insert(originalMap.getDimPosition(idx));
    }
  }

  for (const auto &idxAndValue : llvm::enumerate(originalMap.getResults())) {
    auto idx = idxAndValue.index();
    AffineExpr resultExpr = idxAndValue.value();
    if (resultExpr.isa<AffineDimExpr>()) {
      auto swap1 = originalMap.getDimPosition(idx);
      auto swap2 =
          originalMap.getNumInputs() - originalMap.getNumResults() + idx;
      perm[swap1] = swap2;
      // Only do swap if the output expr does not define another place for the
      // other input dim
      if (!foundInputDims.contains(swap2)) {
        perm[swap2] = swap1;
      }
    }
  }
}

// This function will take a input Value and a index map that represents the
// coordinate mapping that could be a combination of tranposes and broadcasts
// and insert the necessary TransformOps
static Value insertTransposeAndBroadcastTransforms(PatternRewriter &b,
                                                   MemRefType outType,
                                                   Value inp,
                                                   AffineMap inpIdxMap) {
  if (!inpIdxMap.isIdentity()) {
    Location loc = inp.getLoc();
    auto inpType = inp.getType().template cast<MemRefType>();
    ArrayRef<int64_t> inpShape = inpType.getShape();
    ArrayRef<int64_t> outShape = outType.getShape();

    int64_t diff = outShape.size() - inpShape.size();
    LLVM_DEBUG(llvm::dbgs() << "Reached makeBroadcast with map " << inpIdxMap
                            << " and diff = " << diff << "\n");

    SmallVector<uint32_t> bcastDims;
    SmallVector<uint32_t> bcastInDims;
    SmallVector<uint32_t> passThroughInDims;
    SmallVector<uint32_t> perm;
    createPermutationForMinorIdentityWithBroadcast(inpIdxMap, perm);
    auto permMap = AffineMap::getPermutationMap(perm, b.getContext());
    inpIdxMap = inpIdxMap.compose(permMap);
    assert(
        (inpIdxMap.isMinorIdentityWithBroadcasting(&bcastDims)) &&
        "this is guranteed by createPermutationForMinorIdentityWithBroadcast");

    // Broadcast those dimensions that the original linalg.generic map specifies
    // are broadcast and collect their locations, accounting for the leading
    // dimensions not represented in that map but which are present in the gemm
    // coordinates
    BottomUpTMBuilder bcastTransform(b, inpShape, loc);
    bool hasBcast = false;
    for (uint32_t i = 0; i < inpShape.size(); ++i) {
      if (!llvm::is_contained(bcastDims, i)) {
        // Here the diff correspond to leading dropped dimensions when going
        // from output co-ordinates to input co-ordinates.
        assert(inpIdxMap.getDimPosition(i) == diff + i);
        passThroughInDims.push_back(diff + i);
        bcastTransform.passThrough({i}, {i});
      } else {
        hasBcast = true;
        bcastInDims.push_back(diff + i);
        bcastTransform.broadcast({i}, {outShape[diff + i]});
      }
    }
    if (hasBcast) {
      inp = b.create<TransformOp>(loc, inp, bcastTransform.get());
    }

    // Then, add dimensions that are present in the writeback coordinates but
    // are not present in the additional fusion argument with matching sizes.
    // This, combined with the previous step, ensures that the view of the
    // fusion argument has the same dimensions as the gemm output, though they
    // are not necessarily in the same order.
    bool isDimAdded = false;
    BottomUpTMBuilder addDimtransform(
        b, inp.getType().cast<ShapedType>().getShape(), loc);
    for (uint32_t i = 0; i < outShape.size(); ++i) {
      unsigned int startIdx = i - diff;
      if (llvm::is_contained(bcastInDims, i)) {
        addDimtransform.passThrough({i}, {startIdx});
      } else if (llvm::is_contained(passThroughInDims, i)) {
        addDimtransform.passThrough({i}, {startIdx});
      } else {
        isDimAdded = true;
        SmallString<8> name;
        ("exp" + Twine(i)).toVector(name);
        addDimtransform.addDim(name, i, outShape[perm[i]]);
      }
    }
    if (isDimAdded) {
      inp = b.create<TransformOp>(loc, inp, addDimtransform.get());
    }

    // Permute the dimensions of the fusion argument to match those of the gemm
    // writeback by applying the inverse of the permutation that would have made
    // the original indexing map into a minor identity with broadcast. The
    // inverse of that permutation takes the gemm writeback coordinates and
    // scatters them into positions that match the non-identity indexing pattern
    // of the fusion argument.
    if (!permMap.isIdentity()) {
        BottomUpTMBuilder permtransform(b, inp.getType().cast<ShapedType>().getShape(), loc);
        llvm::SmallVector<uint32_t, 4> identityVec;
        for (uint32_t i = 0; i < outShape.size(); ++i) {
          identityVec.push_back(i);
        }
        permtransform.passThrough(identityVec, perm);
        inp = b.create<TransformOp>(loc, inp, permtransform.get());
    }
    return inp;
  }
  return inp;
}

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

  ArrayAttr sourceLeftOob = gemmStoreOp.getLeftOobDims();
  ArrayAttr sourceRightOob = gemmStoreOp.getRightOobDims();

  // In general, note that keeping the vectorization of the writeback is safe
  // on account of the fact that vectorization means that the maps for the
  // gemm output (and thus the extra argument) are contiguous in the
  // underlying memory.

  // If there are no broadcasts, re-use the coordianes for the writeback
  if (sourceTransformsFromOp.empty()) {
    Type typeToLoad = destElemType;
    if (copyLength > 1)
      typeToLoad = VectorType::get({copyLength}, typeToLoad);

    Value loaded = b.create<GlobalLoadOp>(
        loc, typeToLoad, source, sourceLeftOob, sourceRightOob, loadCoord);
    b.create<InBoundsStoreOp>(loc, loaded, dest, zero);
  } else {
    // Note: the vectorization of extra argument may be smaller than the
    // vectorization of the convolution.
    size_t extraMapInSize = loadCoord.size();
    std::tie(sourceLeftOob, sourceRightOob) = computeOobFromTransforms(
        b, sourceTransformsFromOp, {{sourceLeftOob, sourceRightOob}});

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
          loc, typeToLoad, source, sourceLeftOob, sourceRightOob,
          copyLoop.getLowerCoords(/*domain=*/0));
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
  ret = insertTransposeAndBroadcastTransforms(b, outType, ret, outToInpMap);

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
      auto laGenericOutIdxMap =
      laGeneric.getTiedIndexingMap(laGenericOut);
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

void RockLinalgAlignPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<InlineViewLikeOperandsLinalgRewritePattern>(ctx);
  patterns.add<MILARewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
