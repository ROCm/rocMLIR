//===- AlignTiling.cpp - Align Linalg ops with MIOpen ops
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
// based on miopen lowering step2.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/TransformMapBuilder.h"
#include "mlir/Dialect/MIOpen/utility/transformMapUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <numeric>

#define DEBUG_TYPE "miopen-linalg-align"

using namespace mlir;
using namespace mlir::miopen;

namespace {
struct MIOpenLinalgAlignPass
    : public MIOpenLinalgAlignPassBase<MIOpenLinalgAlignPass> {
  void runOnOperation() override;
};

struct MILARewritePattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp laGeneric,
                                PatternRewriter &b) const override;
};
} // end anonymous namespace

/// If `inpMap` is a map of the form
/// (d0, d1, ..., dk) -> (d(i0), d(i1), ..., d(ik))
/// where thi i's don't make it the identity map, wrap `inp` in a
/// miopen.transform that corresponds to the map, and returns the indexing map
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

    // 1. insert a broadcast miopen.transform
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

static void insertCopyFromOtherArg(PatternRewriter &b, Location loc,
                                   ThreadwiseCopyV2Op op, Value srcOp,
                                   Value dest) {
  LLVM_DEBUG(llvm::dbgs() << "Src type: " << srcOp.getType()
                          << " dest type: " << op.dest().getType() << "\n");
  ArrayRef<int64_t> sType, dType;
  sType = srcOp.getType().cast<ShapedType>().getShape();
  dType = op.dest().getType().cast<ShapedType>().getShape();
  assert(sType.size() == dType.size() &&
         "Rank of extra fusion arguments matches shape of C tensor");
  for (unsigned i = 0; i < sType.size(); i++) {
    assert((sType[i] == dType[i] || sType[i] == 1) &&
           "shape of extra fusion arguments matches shape of C tensor or "
           "broadcastable");
  }

  auto writeCLoop = op->getParentOfType<TransformingForOp>();
  assert(writeCLoop && "threadwise_copy_v2 must be in a transforming_for");

  // Handle broadcasts introduced during fusion.
  ArrayAttr sourceTransformsFromOp;
  Value source;
  std::tie(source, sourceTransformsFromOp) = untransform(b, srcOp);

  int64_t copyLength = op.length().getSExtValue();
  Type typeToLoad = dest.getType().cast<MemRefType>().getElementType();
  if (copyLength > 1)
    typeToLoad = VectorType::get({copyLength}, typeToLoad);

  ArrayAttr sourceLeftOob = op.leftOobDims();
  ArrayAttr sourceRightOob = op.rightOobDims();
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);

  // In general, note that keeping the vectorization of the writeback is safe
  // on account of the fact that vectorization means that the maps for the
  // gemm output (and thus the extra argument) are contiguous in the
  // underlying memory.

  // If there are no broadcasts, re-use the coordianes for the writeback
  if (sourceTransformsFromOp.empty()) {
    Value loaded = b.create<BufferLoadOp>(
        loc, typeToLoad, source, sourceLeftOob, sourceRightOob, op.destCoord(),
        /*offset=*/IntegerAttr());
    b.create<InBoundsStoreOp>(loc, loaded, dest, zero);
  } else {
    // Note: this is a hack around the fact that we don't have a good way
    // to add a domain to the enclosing loop currently.
    size_t extraMapInSize = op.destCoord().size();
    SmallVector<int64_t> consts(extraMapInSize, 1LL);
    std::tie(sourceLeftOob, sourceRightOob) = computeOobFromTransforms(
        b, sourceTransformsFromOp, {{sourceLeftOob, sourceRightOob}});

    auto copyLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{op.destCoord()},
        ArrayRef<Attribute>{sourceTransformsFromOp}, /*bounds=*/consts,
        /*strides=*/ArrayRef<int64_t>(consts), /*forceUnroll=*/true,
        /*useIndexDiffs=*/true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(copyLoop.getBody());
      Value loaded = b.create<BufferLoadOp>(
          loc, typeToLoad, source, sourceLeftOob, sourceRightOob,
          copyLoop.getLowerCoords(/*domain=*/0), /*offset=*/IntegerAttr());
      b.create<InBoundsStoreOp>(loc, loaded, dest, zero);
    }
  }
}

static Value makeTransformingCopyLoop(PatternRewriter &b,
                                      ThreadwiseCopyV2Op miTwCopy, Value inp) {
  // 0. capture the memref containing the outputs being written
  Location loc = miTwCopy.getLoc();
  Value gemmOuts = miTwCopy.source();
  auto gemmOutsType = gemmOuts.getType().cast<MemRefType>();
  int64_t sliceLength = miTwCopy.length().getSExtValue();
  auto sliceLengthType = gemmOutsType.clone(sliceLength).cast<MemRefType>();

  // 1. create a second allocation of the same type to hold loaded elements
  Value alloc = b.create<GpuAllocOp>(loc, sliceLengthType);

  // 2. clone twcopy for <addend> -> regs as transforming_for
  insertCopyFromOtherArg(b, loc, miTwCopy, inp, alloc);
  return alloc;
}

Value applyTransforms(PatternRewriter &b, ThreadwiseCopyV2Op miTWCopy,
                      Value inp, AffineMap inpMap) {
  Value ret = inp;
  // 0. move all input preceding ops before
  Operation *pred = miTWCopy;
  while (Operation *op = inp.getDefiningOp()) {
    assert(isa<memref::ExpandShapeOp>(op) || isa<memref::CollapseShapeOp>(op));
    op->moveBefore(pred);
    pred = op;
    inp = op->getOperand(0);
  }

  // 1. insert broadcast op if necessary
  MemRefType outType = miTWCopy.dest().getType().cast<MemRefType>();
  std::tie(ret, inpMap) = makeTransposeTransform(b, ret, inpMap);
  ret = makeBroadcast(b, outType, ret, inpMap);

  // 2. also create threadwise_copy_v2 from global to regs
  //    TODO(sjw): make sure output buffer writes (means these inputs will be
  //    buffer reads)
  return makeTransformingCopyLoop(b, miTWCopy, ret);
}

static ThreadwiseCopyV2Op traceToThreadwiseCopy(Value inp) {
  // 1. Validate that the only uses of the linalg.generic input are the one
  // generic and a copy operation or transform.
  bool allValidUses = true;
  ThreadwiseCopyV2Op result;
  for (Operation *use : inp.getUsers()) {
    if (isa<memref::DeallocOp>(use)) {
      // ignore
      continue;
    }
    if (isa<linalg::GenericOp>(use)) {
      // reader
    } else if (auto copy = dyn_cast<ThreadwiseCopyV2Op>(use)) {
      // Threadwise copy that is already unttransformed (new style)
      if (result) {
        LLVM_DEBUG(llvm::dbgs() << "Multiple copies somehow, no fusion\n");
        allValidUses = false;
      }
      result = copy;
    } else {
      allValidUses = false;
    }
  }

  // Additionally catch the case when gemm result had to be expanded before
  // being fed.
  if (auto expanded = inp.getDefiningOp<memref::ExpandShapeOp>()) {
    auto src = expanded.getSrc();
    for (Operation *use : src.getUsers()) {
      if (auto copy = dyn_cast<ThreadwiseCopyV2Op>(use)) {
        if (result) {
          LLVM_DEBUG(llvm::dbgs() << "Multiple copies somehow, no fusion\n");
          allValidUses = false;
        }
        result = copy;
      }
    }
  }

  if (!result)
    LLVM_DEBUG(llvm::dbgs() << "Align tiling: generic not tracing to copy");
  if (!allValidUses)
    LLVM_DEBUG(llvm::dbgs() << "Align tiling: found invalid use\n");
  return allValidUses ? result : ThreadwiseCopyV2Op();
}

// Returns the value of the buffer that's meant to be the new writeback.
static Value reconfigureLAGeneric(PatternRewriter &b,
                                  linalg::GenericOp laGeneric, Value laIn,
                                  ArrayRef<AffineMap> idxMaps,
                                  ThreadwiseCopyV2Op twcopy) {
  MLIRContext *ctx = laGeneric.getContext();
  Location loc = laGeneric.getLoc();
  Value twout = twcopy.dest();
  auto regType = laIn.getType().template cast<MemRefType>();
  auto laOut = b.create<GpuAllocOp>(loc, regType);

  SmallVector<AffineMap, 5> laGenericAMaps;
  SmallVector<Value, 5> newInputs;
  for (auto pair : llvm::zip(laGeneric.inputs(), idxMaps)) {
    if (Value inp = std::get<0>(pair)) {
      AffineMap inpIdxMap = std::get<1>(pair);
      Value newInput;
      auto expanded = inp.getDefiningOp<memref::ExpandShapeOp>();
      if (inp == twout || (expanded && expanded.getSrc() == twout)) {
        newInput = laIn;
      } else {
        // 2.1.1. Align tiling of other inputs
        newInput = applyTransforms(b, twcopy, inp, inpIdxMap);
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

static bool checkCompatibleTypes(AffineMap inpMap, AffineMap outMap) {
  if (inpMap == outMap && inpMap.isIdentity())
    return true;

  // Failing that, the more complicated check is to ensure the input map
  // can be transformed into the output map.
  if (inpMap.getNumDims() == outMap.getNumDims()) {
    SmallVector<uint32_t, 4> permutation;
    return inpMap.isProjectedPermutation(/*allowZeroInResults=*/true);
  }
  return false;
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

  SmallVector<AffineMap> idxMaps = laGeneric.getIndexingMaps();
  // Output must be indexed by identity map for this to work
  AffineMap outIdxMap = idxMaps.back();
  if (!outIdxMap.isIdentity())
    return failure();

  // 1. Trace input to threadwise_copy. Collect transforms (to be applied to
  // other inputs).
  // 1.1. Find the conv2d output
  ThreadwiseCopyV2Op copyOp;
  for (auto pair : llvm::zip(idxMaps, laGeneric.inputs())) {
    AffineMap inpIdxMap = std::get<0>(pair);
    Value inp = std::get<1>(pair);
    ThreadwiseCopyV2Op maybeCopy = traceToThreadwiseCopy(inp);
    if (maybeCopy) {
      if (copyOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Multiple generic inputs come from writeback\n");
        return failure();
      }
      if (!inpIdxMap.isIdentity()) {
        LLVM_DEBUG(llvm::dbgs() << "Writeback input not read with identity map "
                                   "- earlier lowering should've fixed\n");
        return failure();
      }
      copyOp = maybeCopy;
    } else {
      // Other inputs must have access maps compatible with current fusion
      // rewrites
      if (!checkCompatibleTypes(inpIdxMap, outIdxMap)) {
        LLVM_DEBUG(llvm::dbgs() << "Input index map " << inpIdxMap
                                << " incompatible with output index map "
                                << outIdxMap << "\n");
        return failure();
      }
    }
  }
  if (!copyOp) {
    LLVM_DEBUG(llvm::dbgs() << "Align tiling: couldn't find writeback\n");
    return failure();
  }
  // 2. Apply if input found

  // point back to original memory.
  if (memref::ExpandShapeOp expanded =
          out.getDefiningOp<memref::ExpandShapeOp>()) {
    out = expanded.getSrc();
  }

  Value gemmV2Outs = copyOp.source();
  auto gemmV2OutsType = gemmV2Outs.getType().cast<MemRefType>();
  {
    PatternRewriter::InsertionGuard guard(b);
    // 2.0. Reset insertion point to before the copy.
    b.setInsertionPoint(copyOp);
    Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);

    // 2.1. Take out a slice of the result vector to create a vector-sized
    // slice to enable creating the fusion section.
    int64_t sliceLength = copyOp.length().getSExtValue();
    MemRefType sliceType = gemmV2OutsType.clone(sliceLength).cast<MemRefType>();
    Value fusionSlice = b.create<GpuAllocOp>(loc, sliceType);
    Type typeToCopy = sliceType.getElementType();
    if (sliceType.getNumElements() > 1)
      typeToCopy =
          VectorType::get(sliceType.getShape(), sliceType.getElementType());
    Value sliceVals = b.create<InBoundsLoadOp>(loc, typeToCopy, gemmV2Outs,
                                               copyOp.sourceCoord());
    b.create<InBoundsStoreOp>(loc, sliceVals, fusionSlice, zero);

    // 2.2. Tile linalg.generic with vgpr as input, return output vgprs
    Value laOutRegs =
        reconfigureLAGeneric(b, laGeneric, fusionSlice, idxMaps, copyOp);
    // 2.2.0. Move the generic before the write-back. This'll put all
    // the copy loops for other inputs before the generic due to insertion
    // order.
    laGeneric->moveBefore(copyOp);

    // 2.3. Replace twcopy inputs with la.generic result vgprs

    // Since the threadwise copy arg has gone through untransform()
    // its expected output type is the same as the output type of the
    // linalg.generic.
    copyOp.sourceMutable().assign(laOutRegs);
    // The indexing has been moved into slice creation, reset source
    // coord.
    copyOp.sourceCoordMutable().assign(zero);
    copyOp.destMutable().assign(out);

    return success();
  }

  return failure();
}

void MIOpenLinalgAlignPass::runOnOperation() {
  if (getOperation()->hasAttr("original_func") && !getOperation()->hasAttr("kernel")) return;

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<MILARewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::miopen::createMIOpenLinalgAlignPass() {
  return std::make_unique<MIOpenLinalgAlignPass>();
}
