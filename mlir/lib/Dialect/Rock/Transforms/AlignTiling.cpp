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
llvm::SmallVector<mlir::AffineExpr, 4>
getDelinearizedAffineExpr(mlir::ArrayRef<int64_t> strides,
                          mlir::ArrayRef<int64_t> shapes, Builder &b,
                          unsigned int position) {
  AffineExpr resultExpr = b.getAffineDimExpr(position);
  int64_t rank = strides.size();
  SmallVector<AffineExpr, 4> vectorOffsets(rank);
  // If the rank is 1, expand or collapse shapes will just
  // pass-through the dimensions.
  if (rank == 1) {
    vectorOffsets[0] = resultExpr;
    return vectorOffsets;
  }
  for (unsigned i = 0; i < rank; i++) {
    // If the current shape is 1 and the rank is non-zero,
    // could only mean it is being broadcasted. Hence,
    // putting zero.
    if (shapes[i] == 1) {
      vectorOffsets[i] = resultExpr * 0;
    } else {
      // Recording the vector offsets here.
      vectorOffsets[i] = resultExpr;
      // There is no point of putting a modulo if the size
      // is equivalent to that.
      if (i - 1 >= 0 && shapes[i] != strides[i - 1]) {
        vectorOffsets[i] = vectorOffsets[i] % strides[i - 1];
      }
      vectorOffsets[i] = vectorOffsets[i].floorDiv(strides[i]);

      // The resultExpr has to propagated anyway for
      // other dimensions where the recording in the above
      // will do the neccesary checks to remove the modulo
      if (i - 1 >= 0) {
        resultExpr = resultExpr % strides[i - 1];
      }
      resultExpr = resultExpr.floorDiv(strides[i]);
    }
  }
  return vectorOffsets;
}

// This function will create a affine map that represent the mapping
// from higher rank memref type to lower rank memref type.
static AffineMap createHigherToLowerRankViewAffineMap(PatternRewriter &rewriter,
                                                      ArrayRef<ReassociationIndices> reassociationIndices,
                                                      const MemRefType& higherRankType,
                                                      const MemRefType& lowerRankType){
    SmallVector<AffineExpr, 4> resultExprs;
    int iDimCount = 0;
    for (SmallVector<int64_t, 2> groups :
         reassociationIndices) {
      assert(!groups.empty() && "association indices groups cannot be empty");
      unsigned groupSize = groups.size();
      SmallVector<int64_t> suffixProduct(groupSize);
      // Calculate suffix product for all collapse op source dimension sizes.
      suffixProduct[groupSize - 1] = 1;
      for (unsigned i = groupSize - 1; i > 0; i--)
        suffixProduct[i - 1] =
            suffixProduct[i] * higherRankType.getDimSize(groups[i]);
      // Derive the index values along all dimensions of the source
      // corresponding to the index wrt to collapsed shape op output.
      SmallVector<AffineExpr, 4> srcIndexExpr = getDelinearizedAffineExpr(
          suffixProduct, higherRankType.cast<ShapedType>().getShape(),
          rewriter, iDimCount++);
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
    SmallVector<mlir::ReassociationIndices, 4U> reassociationIndices = collapseOp.getReassociationIndices();
    MemRefType lowerRankType = collapseOp.getType();
    MemRefType higherRankType = collapseOp.getSrcType();
    auto representativeMap = createHigherToLowerRankViewAffineMap(rewriter, reassociationIndices, higherRankType, lowerRankType);
    foldedMap = representativeMap.compose(foldedMap);
    toBeErasedViewLikeOps.push_back(collapseOp);
    return foldViewLikeOperands(rewriter, collapseOp.getViewSource(), foldedMap,
                                rootOp, toBeErasedViewLikeOps);
  }
  if (memref::ExpandShapeOp expandOp =
          op.getDefiningOp<memref::ExpandShapeOp>()) {
    SmallVector<mlir::ReassociationIndices, 4U> reassociationIndices = expandOp.getReassociationIndices();
    MemRefType higherRankType = expandOp.getType();
    MemRefType lowerRankType = expandOp.getSrcType();
    auto representativeMap = createHigherToLowerRankViewAffineMap(rewriter, reassociationIndices, higherRankType, lowerRankType);
    // We take the inverse here because in expand shape it is going from lower to higher rank.
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
  auto loc = laGeneric.getLoc();
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

/// If `inpMap` is a map of the form
/// (d0, d1, ..., dk) -> (d(i0), d(i1), ..., d(ik))
/// where thi i's don't make it the identity map, wrap `inp` in a
/// rock.transform that corresponds to the map, and returns the indexing map
/// that is the result of applying the permutation. If no permutation is needed,
/// returns its inputs.
static std::tuple<Value, AffineMap>
makeTransposeTransform(PatternRewriter &b, Value inp, AffineMap inpMap) {
  if (!inpMap.isMinorIdentityWithBroadcasting()) {
    // permumation[i] says where the map output i should be sent.
    SmallVector<uint32_t> permutation;
    if (inpMap.isPermutationOfMinorIdentityWithBroadcasting(permutation)) {
      Location loc = inp.getLoc();
      MemRefType inputType = inp.getType().cast<MemRefType>();
      ArrayRef<int64_t> inputShape = inputType.getShape();
      LLVM_DEBUG(llvm::dbgs()
                 << "Transpose input type : " << inputType << "\n");
      BottomUpTMBuilder permuteBuilder(b, inputShape, loc);

      SmallVector<uint32_t> identityIdxs;
      identityIdxs.reserve(inputShape.size());
      for (uint32_t idx = 0, e = inputShape.size(); idx < e; ++idx)
        identityIdxs.push_back(idx);

      permuteBuilder.passThrough(permutation, identityIdxs);
      TransformMapAttr permuteAttr = permuteBuilder.get();
      Value ret = b.create<TransformOp>(loc, inp, permuteAttr);
      AffineMap composed = permuteAttr.getMap().getAffineMap().compose(inpMap);
      LLVM_DEBUG(llvm::dbgs() << "indexing = " << inpMap << " then transform "
                              << permuteAttr.getMap().getAffineMap() << " is "
                              << composed << "\n");
      return {ret, composed};
    }
  }
  return {inp, inpMap};
}

static Value makeBroadcast(PatternRewriter &b, MemRefType outType, Value inp,
                           AffineMap inpIdxMap) {
  if (!inpIdxMap.isIdentity()) {
    Location loc = inp.getLoc();
    auto inpType = inp.getType().template cast<MemRefType>();
    ArrayRef<int64_t> inpShape = inpType.getShape();
    ArrayRef<int64_t> outShape = outType.getShape();

    uint32_t diff = outShape.size() - inpShape.size();
    SmallVector<uint32_t> bcastDims;
    LLVM_DEBUG(llvm::dbgs() << "Reached makeBroadcast with map " << inpIdxMap
                            << " and diff = " << diff << "\n");
    if (diff) {
      // 0.1 expand dims (size = 1) in front
      SmallVector<uint32_t, 8> endDims;
      SmallVector<uint32_t, 8> startDims;
      for (uint32_t i = 0, e = inpShape.size(); i < e; ++i) {
        startDims.push_back(i);
        endDims.push_back(inpIdxMap.getDimPosition(i));
      }
      BottomUpTMBuilder transform(b, inpShape, loc);
      transform.passThrough(endDims, startDims);
      for (uint32_t i = 0; i < outShape.size(); ++i) {
        uint32_t *it = llvm::find(endDims, i);
        if (it != endDims.end())
          continue;
        SmallString<8> name;
        ("exp" + Twine(i)).toVector(name);
        transform.addDim(name, i, 1);
        bcastDims.push_back(i);
      }

      inp = b.create<TransformOp>(loc, inp, transform.get());

      inpType = inp.getType().template cast<MemRefType>();
      inpShape = inpType.getShape();
    } else {
      inpIdxMap.isMinorIdentityWithBroadcasting(&bcastDims);
      // Check if it's transposed.
      if (bcastDims.size() == 0)
        return inp;
      LLVM_DEBUG(llvm::dbgs() << "Broadcast dims: ");
      LLVM_DEBUG(llvm::interleaveComma(bcastDims, llvm::dbgs()));
      LLVM_DEBUG(llvm::dbgs() << "\n");
    }

    // 1. insert a broadcast rock.transform
    SmallVector<uint32_t, 8> ptDims;
    SmallVector<int64_t, 8> bcastSizes;
    for (uint32_t dim = 0; dim < inpShape.size(); ++dim) {
      if (std::find(bcastDims.begin(), bcastDims.end(), dim) !=
          bcastDims.end()) {
        bcastSizes.push_back(outShape[dim]);
      } else {
        ptDims.push_back(dim);
      }
    }
    BottomUpTMBuilder transform(b, inpShape, loc);
    transform.passThrough(ptDims, ptDims);
    transform.broadcast(bcastDims, bcastSizes);

    inp = b.create<TransformOp>(loc, inp, transform.get());
  }
  return inp;
}

static void insertLoadFromOtherSource(PatternRewriter &b, Location loc,
                                      GlobalStoreOp gemmStoreOp, Value srcOp,
                                      Value dest) {
  LLVM_DEBUG(llvm::dbgs() << "Src type: " << srcOp.getType() << " dest type: "
                          << gemmStoreOp.getDest().getType() << "\n");
  ArrayRef<int64_t> sType, dType;
  sType = srcOp.getType().cast<ShapedType>().getShape();
  dType = gemmStoreOp.getDest().getType().getShape();
  assert(sType.size() == dType.size() &&
         "Rank of extra fusion arguments matches shape of C tensor");
  SmallVector<Value, 6> loadCoord = gemmStoreOp.getDestCoord();
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  for (unsigned i = 0; i < sType.size(); i++) {
    assert((sType[i] == dType[i] || sType[i] == 1) &&
           "shape of extra fusion arguments matches shape of C tensor or "
           "broadcastable");
    // broadcast source.
    if (sType[i] != dType[i])
      loadCoord[i] = zero;
  }

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
  std::tie(ret, outToInpMap) = makeTransposeTransform(b, ret, outToInpMap);
  ret = makeBroadcast(b, outType, ret, outToInpMap);

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

  SmallVector<AffineMap, 5> laGenericAMaps;
  SmallVector<Value, 5> newInputs;
  for (auto pair : llvm::zip(laGeneric.inputs(), idxMaps)) {
    if (Value inp = std::get<0>(pair)) {
      AffineMap inpIdxMap = std::get<1>(pair);
      auto invertOutIdxMap = inversePermutation(outIdxMap);
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

  SmallVector<AffineMap> idxMaps = laGeneric.getIndexingMapsArray();
  AffineMap outIdxMap = idxMaps.back();

  // 1. Trace input to global_store.
  // 1.1. Find the (implicit) gemm output
  GlobalStoreOp gemmStoreOp;
  for (auto pair : llvm::zip(idxMaps, laGeneric.inputs())) {
    AffineMap inpIdxMap = std::get<0>(pair);
    auto invertOutIdxMap = inversePermutation(outIdxMap);
    auto outToInMap = inpIdxMap.compose(invertOutIdxMap);
    Value inp = std::get<1>(pair);
    GlobalStoreOp maybeStore = traceToGlobalStore(inp);
    if (maybeStore) {
      if (gemmStoreOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Multiple generic inputs come from writeback\n");
        return failure();
      }
      SmallVector<unsigned> broadcastedDims;
      // This is not to allow broadcasting but due to canonical linalg
      // form if the unit dims can carry affine expr to be zero in the
      // translation, hence there is a following check.
      if (!outToInMap.isMinorIdentityWithBroadcasting(&broadcastedDims)) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "The store is not even a minor identity with broadcasting.\n");
        return failure();
      }
      auto inpShape = inp.getType().cast<ShapedType>().getShape();
      for (auto bDim : broadcastedDims) {
        if (inpShape[bDim] != 1) {
          LLVM_DEBUG(llvm::dbgs()
                     << "The store input cannot be a real broacast.\n");
          return failure();
        };
      }
      gemmStoreOp = maybeStore;
    } else {
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
