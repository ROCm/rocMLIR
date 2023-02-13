//===- FoldTranspose.cpp - rewrites to allow Rock kernel fusion  ------===//
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
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKFOLDTRANSPOSEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-fold-transpose"

using namespace mlir;
using namespace mlir::rock;

namespace {

// This rewrite will rewrite the linalg IO that has view like-ops surrounding
// them to be consumed by the linalg operation itself adjusting the indexing
// maps to faithfully represent them.
struct InlineViewLikeOperandsLinalgRewritePattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp laGeneric,
                                PatternRewriter &rewriter) const override;
};

// This function will take strides of the original layout of corresponding
// dimensions being reassociated and the expected resultant dim sizes of
// post-reassociation --> then, produce the set of resultant AffineExprs after
// the reassociation.
static void getDelinearizedAffineExpr(
    ArrayRef<int64_t> originalStridesDimsBeingReassociated,
    ArrayRef<int64_t> postReassociatedDimSizes, Builder &b,
    unsigned int position, SmallVectorImpl<AffineExpr> &res) {
  AffineExpr resultExpr = b.getAffineDimExpr(position);
  int64_t rank = originalStridesDimsBeingReassociated.size();
  // If the rank is 1, expand or collapse shapes will just
  // pass-through the dimensions.
  if (rank == 1) {
    res[0] = resultExpr;
    return;
  }
  for (int i = 0; i < rank; i++) {
    // If the postReassociatedDimSize is 1 and the rank is non-zero,
    // could only mean it is being broadcasted. Hence,
    // putting zero.
    if (postReassociatedDimSizes[i] == 1) {
      res[i] = resultExpr * 0;
    } else {
      // Recording the vector offsets here.
      res[i] = resultExpr;
      // There is no point of putting a modulo if the size
      // is equivalent to that.
      if (i - 1 >= 0 && postReassociatedDimSizes[i] !=
                            originalStridesDimsBeingReassociated[i - 1]) {
        res[i] = res[i] % originalStridesDimsBeingReassociated[i - 1];
      }

      if (postReassociatedDimSizes[i] >
          originalStridesDimsBeingReassociated[i]) {
        // We only need the floorDiv if the dimSize
        // is larger than the stride
        res[i] = res[i].floorDiv(originalStridesDimsBeingReassociated[i]);
      } else if (postReassociatedDimSizes[i] <
                 originalStridesDimsBeingReassociated[i]) {
        // if the shape is smaller than the stride
        // expr might as well be zero.
        res[i] = res[i] * 0;
      }

      // The resultExpr has to propagated anyway for
      // other dimensions where the recording in the above
      // will do the neccesary checks to remove the modulo
      if (i - 1 >= 0) {
        resultExpr = resultExpr % originalStridesDimsBeingReassociated[i - 1];
      }
      resultExpr = resultExpr.floorDiv(originalStridesDimsBeingReassociated[i]);
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
  if (reassociationIndices.empty()) {
    resultExprs = SmallVector<AffineExpr, 4>(higherRankType.getRank(),
                                             rewriter.getAffineConstantExpr(0));
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
    SmallVector<ReassociationIndices, 4U> reassociationIndices =
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
    SmallVector<ReassociationIndices, 4U> reassociationIndices =
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
struct RockFoldTransposePass
    : public rock::impl::RockFoldTransposePassBase<RockFoldTransposePass> {
  void runOnOperation() override;
};
} // end namespace

/// If there is a chain of operations that leads from `v` to
/// a gemm-wrapping op, return that op.
static Operation *getGemmWrapperUser(Value v) {
  for (Operation *user : v.getUsers()) {
    if (isa<RockGemmWrapperInterface>(user))
      return user;
    if (auto transform = dyn_cast<TransformOp>(user))
      if (Operation *upstream = getGemmWrapperUser(transform.getOutput()))
        return upstream;
  }
  return nullptr;
}

namespace {
// MITPRewritePattern
// Fold linarg.generic and memref.alloc generated by transpose op into
// rock.transform
struct RemoveTrivialTransposePattern
    : public OpRewritePattern<linalg::GenericOp> {
  // Explicit constructor to set a higher pattern benefic than the more general
  // pattern below.
  explicit RemoveTrivialTransposePattern(MLIRContext *ctx)
      : OpRewritePattern<linalg::GenericOp>(ctx, /*benefit=*/2) {}

  rock::TransformOp makeTranspose(PatternRewriter &b, Value inp,
                                  const AffineMapAttr &inMap,
                                  const AffineMapAttr &outMap) const {
    AffineMap inpIdxMap = inMap.getAffineMap();
    AffineMap outpIdxMap = outMap.getAffineMap();
    Location loc = inp.getLoc();
    MemRefType inpType = inp.getType().template cast<MemRefType>();
    ArrayRef<int64_t> inpShape = inpType.getShape();

    SmallVector<uint32_t, 8> endDims;
    SmallVector<uint32_t, 8> startDims;
    for (uint32_t i = 0, e = inpShape.size(); i < e; ++i) {
      startDims.push_back(i);
      uint32_t inMapped = inpIdxMap.getDimPosition(i);
      endDims.push_back(outpIdxMap.getDimPosition(inMapped));
    }
    rock::BottomUpTMBuilder transform(b, inpShape, loc);
    transform.passThrough(endDims, startDims);
    auto tfOp = b.create<rock::TransformOp>(loc, inp, transform.get());
    return tfOp;
  }

  LogicalResult matchAndRewrite(linalg::GenericOp laGeneric,
                                PatternRewriter &b) const override {
    // 0. Test compatibility
    // 0.0. Only fully parallel for now
    for (StringRef itr :
         laGeneric.iterator_types().getAsValueRange<StringAttr>()) {
      if (itr != "parallel") {
        return failure();
      }
    }

    bool bPassing = false;
    laGeneric.getRegion().walk([&](linalg::YieldOp yieldOp) {
      Value laReturn = yieldOp->getOperand(0);
      bPassing = (laReturn == laGeneric.getRegion().getArgument(0));
    });

    // 0.1. Test it only passes through 1:1 and no other calculation
    if (laGeneric.inputs().size() != 1 || laGeneric.outputs().size() != 1 ||
        !bPassing) {
      return failure();
    }

    // 0.2. linalg.generic lowered from tosa.transpose should have memref.alloc
    Value out = *laGeneric.outputs().begin();
    auto allocToDel = out.getDefiningOp<memref::AllocOp>();
    if (!allocToDel) {
      return failure();
    }

    // get maps to construct a transforming map for the transpose
    auto idxMaps =
        laGeneric->template getAttrOfType<ArrayAttr>("indexing_maps");
    AffineMapAttr inIdxMap = idxMaps[0].cast<AffineMapAttr>();
    AffineMapAttr outIdxMap = idxMaps[1].cast<AffineMapAttr>();
    auto tpTransform =
        makeTranspose(b, laGeneric->getOperand(0), inIdxMap, outIdxMap);

    b.replaceOp(allocToDel, {tpTransform});
    b.eraseOp(laGeneric);
    return success();
  }
};

// If the input to a linalg.generic is the output of a rock compute op and the
// indexing map for that input is a non-trivial permutation of an identity,
// convert that indexing map to a transpose, expansions and/or collapses. This
// must happen before gridwise gemm conversion because all the transforms on the
// rock compute op output are collected at that time.
struct FoldRockOutputTransforms : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp laGeneric,
                                PatternRewriter &b) const override {
    // We do this out-of-line so as to not invalidate our iterator
    SmallVector<std::tuple<unsigned, Value, AffineMap>> toReplace;

    auto laGenericOut = laGeneric.getOutputOperand(0);
    auto laGenericOutIdxMap = laGeneric.getTiedIndexingMap(laGenericOut);

    for (OpOperand &operand : laGeneric->getOpOperands()) {
      Value opValue = operand.get();
      Operation *convUser = getGemmWrapperUser(opValue);
      if (!convUser)
        continue;
      // If the laGenericOut and rockOp user has the same type
      // there is nothing to be done here.
      if (operand.get().getType() == laGenericOut->get().getType()) {
        continue;
      }

      for (Operation *user : opValue.getUsers()) {
        if (isa<linalg::GenericOp>(user) && user != laGeneric) {
          LLVM_DEBUG(llvm::dbgs() << "Multiple generics on same conv output\n");
          return failure();
        }
      }

      if (!isa_and_nonnull<memref::AllocOp>(opValue.getDefiningOp()))
        continue;

      AffineMap inpIdxMap = laGeneric.getTiedIndexingMap(&operand);
      auto invertInpIdxMap = inversePermutation(inpIdxMap);
      auto inToOutIdxMap = laGenericOutIdxMap.compose(invertInpIdxMap);
      SmallVector<uint32_t, 4> permutation;
      if (!inToOutIdxMap.isPermutationOfMinorIdentityWithBroadcasting(
              permutation))
        continue;

      unsigned opIndex = operand.getOperandNumber();
      toReplace.emplace_back(opIndex, opValue, inToOutIdxMap);
    }

    // Actually do the rewrites, if any
    SmallVector<AffineMap> idxMaps = laGeneric.getIndexingMapsArray();
    SmallVector<Value> inputs = laGeneric.inputs();
    bool hasReplaced = false;
    for (auto &tuple : toReplace) {
      unsigned opIndex = std::get<0>(tuple);
      Value opValue = std::get<1>(tuple);
      AffineMap idxMap = std::get<2>(tuple);
      auto allocation = cast<memref::AllocOp>(opValue.getDefiningOp());
      // Swap out the allocation for the form it needs to take in order to
      // eliminate the non-trivial map.
      auto newShape =
          laGenericOut->get().getType().cast<ShapedType>().getShape();

      // All this new stuff needs to go where the old memref.alloc was
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointAfterValue(allocation);
      auto newAllocType =
          MemRefType::get(newShape, allocation.getType().getElementType());
      MemRefType outType = allocation.getType();
      Value newAlloc =
          b.create<memref::AllocOp>(allocation.getLoc(), newAllocType);
      Value transformedNewAlloc = insertTransposeAndBroadcastTransforms(
          b, outType.getShape(), newAlloc, idxMap);
      if (transformedNewAlloc == newAlloc) {
        // No transform needed, remove the unecessary allocation
        b.eraseOp(newAlloc.getDefiningOp());
        continue;
      } else {
        hasReplaced = true;
      }

      llvm::SmallPtrSet<Operation *, 2> skips = {
          laGeneric, transformedNewAlloc.getDefiningOp()};
      opValue.replaceAllUsesExcept(transformedNewAlloc, skips);

      // Correct indexing maps and changing the inputs
      idxMaps[opIndex] = laGenericOutIdxMap;
      inputs[opIndex] = newAlloc;
    }
    laGeneric->setAttr(laGeneric.indexing_mapsAttrName(),
                       b.getAffineMapArrayAttr(idxMaps));
    laGeneric.getInputsMutable().assign(inputs);
    return success(hasReplaced);
  }
};
} // end namespace

void RockFoldTransposePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto func = getOperation();
  if (!func->hasAttr("kernel")) {
    return;
  }

  RewritePatternSet patternsTP(ctx);
  patternsTP.add<RemoveTrivialTransposePattern,
                 InlineViewLikeOperandsLinalgRewritePattern,
                 FoldRockOutputTransforms>(ctx);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patternsTP))))
    signalPassFailure();
}
