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

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/TransformMapBuilder.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

#include <numeric>

using namespace mlir;

namespace {
struct MIOpenLinalgAlignPass
    : public MIOpenLinalgAlignPassBase<MIOpenLinalgAlignPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

//===- MILARewritePattern -------------------------------------------------===//
//===-  ------------------------------------------------===//
template <typename T> struct MILARewritePattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  mutable llvm::SmallDenseSet<Operation *> processSet;

  template <typename Top> Value getOpResult(OpOperand &use) const {
    Value v;
    auto *ownr = use.getOwner();
    if (auto op = dyn_cast<Top>(ownr)) {
      v = op->getResult(0);
    }
    return v;
  }

  template <typename Top> Value backtrace(Value inv) const {
    if (inv.hasOneUse()) {
      return getOpResult<Top>(*inv.use_begin());
    }
    return Value();
  }

  template <typename Top> Value getOpOperand(OpOperand &use, int idx) const {
    Value v;
    auto *ownr = use.getOwner();
    if (auto op = dyn_cast<Top>(ownr)) {
      v = op->getOperand(idx);
    }
    return v;
  }

  template <typename Top> Value backtrace(Value inv, int idx) const {
    if (inv.hasOneUse()) {
      return getOpOperand<Top>(*inv.use_begin(), idx);
    }
    return Value();
  }

  typedef SmallVector<miopen::TransformOp, 4> TransformList;

  Value bwTraceTo(Value inp, TransformList *transforms = nullptr) const {
    if (auto bt = inp.getDefiningOp<miopen::TransformOp>()) {
      if (transforms)
        transforms->push_back(bt);
      return bwTraceTo(bt->getOperand(0), transforms);
    }
    if (auto bt = inp.getDefiningOp<miopen::GpuAllocOp>())
      return inp;
    if (auto bt = inp.getDefiningOp<memref::AllocOp>())
      return inp;
    return Value();
  }

  Value makeThreadwiseCopy(PatternRewriter &b, Operation *miTWCopy,
                           Value inp) const {
    // 0. capture reg alloc for miTWCopy output regs
    TransformList transforms;
    auto twinp = miTWCopy->getOperand(0);
    // auto miTransform = twinp.getDefiningOp<miopen::TransformOp>();
    // auto miTransformInp = miTransform->getOperand(0);
    // auto miAlloc = miTransformInp.getDefiningOp<miopen::GpuAllocOp>();
    auto miAlloc = bwTraceTo(twinp, &transforms);

    // 1. clone reg alloc + transform
    BlockAndValueMapping cloningMap;
    auto nAlloc = b.clone(*miAlloc.getDefiningOp(), cloningMap);
    Operation *nTransform = nAlloc;
    for (auto transform : llvm::reverse(transforms)) {
      cloningMap.map(transform->getOperand(0), nTransform->getResult(0));
      nTransform = b.clone(*transform, cloningMap);
    }

    // 2. clone twcopy for <addend> -> regs
    cloningMap.map(miTWCopy->getOperand(0), inp);
    cloningMap.map(miTWCopy->getOperand(1), nTransform->getResult(0));
    auto nTWCopy = b.clone(*miTWCopy, cloningMap);

    // 3. swap input coords with output coords
    auto shape = twinp.getType().cast<ShapedType>().getShape();
    for (uint i = 0; i < shape.size(); ++i) {
      uint inIdx = 2 + i;
      uint outIdx = 2 + shape.size() + i;
      auto inCoord = nTWCopy->getOperand(inIdx);
      auto outCoord = nTWCopy->getOperand(outIdx);
      nTWCopy->setOperand(inIdx, outCoord);
      nTWCopy->setOperand(outIdx, inCoord);
    }

    // 4. add bound attr with the register dims
    SmallVector<Attribute, 2> twCopyBoundsAttr;
    for (auto v : shape) {
      twCopyBoundsAttr.push_back(b.getIndexAttr(v));
    }
    nTWCopy->setAttr("bounds", b.getArrayAttr(twCopyBoundsAttr));

    return nAlloc->getResult(0);
  }

  Value makeBroadcast(PatternRewriter &b, MemRefType outType, Value inp,
                      const AffineMapAttr &inpMap) const {
    auto inpIdxMap = inpMap.getAffineMap();
    if (!inpIdxMap.isIdentity()) {
      auto loc = inp.getLoc();
      auto inpType = inp.getType().template cast<MemRefType>();
      auto inpShape = inpType.getShape();
      auto outShape = outType.getShape();

      uint32_t diff = outShape.size() - inpShape.size();
      SmallVector<uint32_t> bcastDims;
      if (diff) {
        // 0.1 expand dims (size = 1) in front
        SmallVector<uint32_t, 8> endDims;
        SmallVector<uint32_t, 8> startDims;
        for (uint32_t i = 0, e = inpShape.size(); i < e; ++i) {
          startDims.push_back(i);
          endDims.push_back(i + diff);
        }
        miopen::BottomUpTMBuilder transform(b, inpShape, loc);
        transform.passThrough(endDims, startDims);
        for (uint32_t i = 0; i < diff; ++i) {
          SmallString<8> name;
          ("exp" + Twine(i)).toVector(name);
          transform.addDim(name, i, 1);
          bcastDims.push_back(i);
        }

        inp = b.create<miopen::TransformOp>(loc, inp, transform.get(),
                                            inpType.getMemorySpaceAsInt());

        inpType = inp.getType().template cast<MemRefType>();
        inpShape = inpType.getShape();
      } else {
        inpIdxMap.isMinorIdentityWithBroadcasting(&bcastDims);
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
      miopen::BottomUpTMBuilder transform(b, inpShape, loc);
      transform.passThrough(ptDims, ptDims);
      transform.broadcast(bcastDims, bcastSizes);

      inp = b.create<miopen::TransformOp>(loc, inp, transform.get(),
                                          inpType.getMemorySpaceAsInt());
    }
    return inp;
  }

  Value createCollapseShapeOp(PatternRewriter &b, Location loc,
                              Value source) const {
    auto ctx = b.getContext();
    auto sourceType = source.getType().cast<ShapedType>();
    assert(sourceType.hasStaticShape() &&
           "Only memrefs with static shapes are allowed");

    auto shape = sourceType.getShape();
    uint64_t collapsedDim = 1;
    SmallVector<AffineExpr, 2> exprs;
    for (uint32_t dim = 0; dim < shape.size(); ++dim) {
      collapsedDim *= shape[dim];
      exprs.push_back(getAffineDimExpr(dim, ctx));
    }

    SmallVector<int64_t, 1> collapsedShape;
    SmallVector<ReassociationExprs, 1> reassocs;
    collapsedShape.push_back(collapsedDim);
    reassocs.push_back(exprs);

    auto collapsedType =
        MemRefType::get(collapsedShape, sourceType.getElementType(), {}, 5);
    Value result =
        b.create<memref::CollapseShapeOp>(loc, collapsedType, source, reassocs);
    return result;
  }

  miopen::TransformingForOp makeLoadToVector(PatternRewriter &b, Location loc,
                                             Operation *miTWCopy, Value srcOp,
                                             Value dstOp) const {
    auto op = dyn_cast<miopen::ThreadwiseCopyV2Op>(miTWCopy);
    auto inShape = srcOp.getType().cast<ShapedType>().getShape();
    SmallVector<Value, 6> inCoords, outCoords;
    for (uint i = 0; i < inShape.size(); ++i) {
      uint inIdx = 2 + i;
      uint outIdx = 2 + inShape.size() + i;
      auto inCoord = op->getOperand(inIdx);
      auto outCoord = op->getOperand(outIdx);
      inCoords.push_back(outCoord);
      outCoords.push_back(inCoord);
    }

    ArrayAttr sourceTransformsOnOp = op.transforms()[1].cast<ArrayAttr>();
    ArrayAttr destTransformsOnOp = op.transforms()[0].cast<ArrayAttr>();
    ArrayAttr sourceTransforms, destTransforms;

    Value source, dest;
    std::tie(source, sourceTransforms) =
        miopen::untransform(b, srcOp, sourceTransformsOnOp);
    std::tie(dest, destTransforms) =
        miopen::untransform(b, dstOp, destTransformsOnOp);
    SmallVector<int64_t, 6> bounds;
    llvm::transform(op.bounds().getAsRange<IntegerAttr>(),
                    std::back_inserter(bounds),
                    [](const IntegerAttr &v) -> int64_t { return v.getInt(); });

    int64_t dataPerCopy =
        op->getAttrOfType<IntegerAttr>("data_per_copy").getInt();

    auto loadType = dstOp.getType().dyn_cast<MemRefType>();
    auto shape = loadType.cast<ShapedType>().getShape();
    auto vecType = VectorType::get(dataPerCopy, loadType.getElementType());
    bounds[shape.size() - 1] /= dataPerCopy;

    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    ArrayAttr srcLeftOob, srcRightOob;
    std::tie(srcLeftOob, srcRightOob) =
        miopen::computeOobFromTransforms(b, sourceTransforms);

    miopen::TransformingForOp copyLoop = b.create<miopen::TransformingForOp>(
        loc, ArrayRef<ValueRange>{inCoords, outCoords},
        ArrayRef<Attribute>{sourceTransforms, destTransforms}, bounds,
        /*forceUnroll=*/true, /*useIndexDiffs=*/true);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(copyLoop.getBody());

    Value loaded;
    if (dataPerCopy > 1) {
      loaded = b.create<miopen::BufferLoadOp>(
          loc, vecType, source, srcLeftOob, srcRightOob,
          copyLoop.getLowerCoords(/*domain=*/0));
    } else {
      loaded = b.create<miopen::BufferLoadOp>(
          loc, loadType.getElementType(), source, srcLeftOob, srcRightOob,
          copyLoop.getLowerCoords(/*domain=*/0));
    }
    SmallVector<Value, 6> indicies;
    for (uint i = 0; i < shape.size() - 1; ++i) {
      indicies.push_back(c0);
    }
    indicies.push_back(copyLoop.getLowerCoords(/*domain=*/1)[0]);
    b.create<miopen::InBoundsStoreOp>(loc, loaded, dest, indicies);
    return copyLoop;
  }

  Value makeTransformingCopy(PatternRewriter &b, Operation *miTWCopy,
                             Value inp) const {
    // 0. capture the slice of vector for miTWCopy output regs
    auto twinp = miTWCopy->getOperand(0);
    auto miVecSlice = twinp.getDefiningOp<miopen::ExtractSliceOp>();

    // 1. clone vector into reg alloc
    BlockAndValueMapping cloningMap;
    auto nVecSlice = b.clone(*miVecSlice, cloningMap);
    auto loc = nVecSlice->getLoc();
    auto regVecType = miVecSlice.getType().template cast<VectorType>();
    auto shape = inp.getType().cast<ShapedType>().getShape();
    SmallVector<int64_t, 6> regShape;
    SmallVector<Value, 6> indexVector;
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    for (uint i = 0; i < shape.size(); ++i) {
      regShape.push_back(1);
      indexVector.push_back(c0);
    }
    regShape[shape.size() - 1] = regVecType.getNumElements();
    auto elemType = regVecType.getElementType();
    auto regType = MemRefType::get(regShape, elemType, {}, 5);
    auto clonedVec = b.create<miopen::GpuAllocOp>(loc, regType);
    auto indices = ValueRange(indexVector);
    b.create<vector::StoreOp>(loc, nVecSlice->getResult(0), clonedVec, indices);

    // 2. clone twcopy for <addend> -> regs as transforming_for
    makeLoadToVector(b, loc, miTWCopy, inp, clonedVec->getResult(0));

    // 3. shrink the dim back to original so it can match the linalg dimensions
    auto cvCollapsed = createCollapseShapeOp(b, loc, clonedVec->getResult(0));
    return cvCollapsed;
  }

  Value applyTransforms(PatternRewriter &b, Operation *miTWCopy, Value inp,
                        const AffineMapAttr &inpMap,
                        SmallVector<Value, 5> &transforms) const {
    Value ret = inp;

    // 0. move all input preceding ops before
    Operation *pred = miTWCopy;
    while (Operation *op = inp.getDefiningOp()) {
      assert(isa<memref::ExpandShapeOp>(op) ||
             isa<memref::CollapseShapeOp>(op));
      op->moveBefore(pred);
      pred = op;
      inp = op->getOperand(0);
    }

    // 1. insert broadcast op if necessary
    assert(transforms.size());
    auto miTransform0 = transforms[0].getDefiningOp<miopen::TransformOp>();
    assert(miTransform0);
    auto outType =
        miTransform0.getOperand().getType().template cast<MemRefType>();
    ret = makeBroadcast(b, outType, ret, inpMap);

    BlockAndValueMapping cloningMap;
    // 2. clone the same transforms applied to the output memory and
    //    apply to all other inputs to the linalg.generic
    for (auto transform : transforms) {
      assert(transform.hasOneUse());
      Operation *tcopy;
      auto miTransform = transform.getDefiningOp<miopen::TransformOp>();
      assert(miTransform);
      cloningMap.map(miTransform->getOperand(0), ret);
      tcopy = b.clone(*miTransform, cloningMap);
      ret = tcopy->getResult(0);
    }

    // 2. also create threadwise_copy from global to regs
    //    TODO(sjw): make sure output buffer writes (means these inputs will be
    //    buffer reads)
    if (auto copyType = dyn_cast<miopen::ThreadwiseCopyOp>(miTWCopy)) {
      return makeThreadwiseCopy(b, miTWCopy, ret);
    } else {
      return makeTransformingCopy(b, miTWCopy, ret);
    }
  }

  template <typename Ttwcopy>
  Value traceToThreadwiseCopy(Value inp,
                              SmallVector<Value, 5> &transforms) const {
    Value null;
    Value nextVal;
    // 1. get reader (memref.expand_shape), return result
    int cnt = 0;
    for (auto &use : inp.getUses()) {
      if (auto op = dyn_cast<linalg::GenericOp>(use.getOwner())) {
        // reader
      } else if (!nextVal) {
        nextVal = getOpResult<miopen::TransformOp>(use);
        transforms.push_back(nextVal);
      }
      cnt++;
    }
    if (cnt == 2) {
      while (nextVal) {
        // 2. get reader (miopen.transform), return result
        if (auto transform = backtrace<miopen::TransformOp>(nextVal)) {
          nextVal = transform;
          transforms.push_back(nextVal);
        } else {
          // 3. get readers (miopen.threadwise_copy)
          bool allTWCopy = true;
          for (auto &use : nextVal.getUses()) {
            allTWCopy &= isa<Ttwcopy>(use.getOwner());
          }
          if (allTWCopy)
            return inp;
          break;
        }
      }
    }
    transforms.clear();
    return null;
  }

  Value reconfigureLAGeneric(T &laGeneric, PatternRewriter &b,
                             SmallVector<Value, 5> &transforms, Value laIn,
                             Operation *twcopy) const {
    auto ctx = laGeneric.getContext();
    auto loc = laGeneric.getLoc();
    Value twout = bwTraceTo(twcopy->getOperand(1));
    auto regType = laIn.getType().template cast<MemRefType>();
    auto laOut = b.create<miopen::GpuAllocOp>(loc, regType);

    auto idxMaps =
        laGeneric->template getAttrOfType<ArrayAttr>("indexing_maps");

    SmallVector<AffineMap, 5> laGenericAMaps;
    SmallVector<Value, 5> newInputs;
    for (auto pair : llvm::zip(laGeneric.inputs(), idxMaps)) {
      if (auto inp = std::get<0>(pair)) {
        auto inpIdxMap = std::get<1>(pair).template cast<AffineMapAttr>();
        Value newInput;
        if (inp == twout) {
          newInput = laIn;
        } else {
          // 2.1.1. Align tiling of other inputs
          newInput = applyTransforms(b, twcopy, inp, inpIdxMap, transforms);
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
    SmallVector<StringAttr, 5> laGenericIteratorArr(
        regType.getRank(), b.getStringAttr("parallel"));
    laGeneric.iterator_typesAttr(b.getArrayAttr(ArrayRef<Attribute>(
        laGenericIteratorArr.begin(), laGenericIteratorArr.end())));
    return laOut;
  }

  static bool checkCompatibleTypes(Type inpType, const AffineMapAttr &inpMap,
                                   Type outType, const AffineMapAttr &outMap) {
    auto inpIdxMap = inpMap.getAffineMap();
    if (inpType == outType && inpMap == outMap && inpIdxMap.isIdentity()) {
      return true;
    } else {
      // check inpMap has same idx input shape
      auto outIdxMap = outMap.getAffineMap();
      if (inpIdxMap.getNumDims() == outIdxMap.getNumDims()) {
        if (inpIdxMap.isMinorIdentity() ||
            inpIdxMap.isMinorIdentityWithBroadcasting())
          return true;
      }
    }
    return false;
  }

  LogicalResult matchAndRewrite(T laGeneric,
                                PatternRewriter &b) const override {
    LogicalResult fail = failure();
    auto loc = laGeneric.getLoc();

    if (processSet.find(laGeneric) != processSet.end()) {
      return fail;
    }
    processSet.insert(laGeneric);

    // 0. Test compatibility
    // 0.0. Only fully parallel for now
    for (auto itr : laGeneric.iterator_types()) {
      if (itr.template cast<StringAttr>().getValue() != "parallel") {
        return fail;
      }
    }

    Value out = *laGeneric.outputs().begin(); // may be another arg
    // 0.1. Test compatibility,  Only 1 output supported
    if (laGeneric.outputs().size() > 1) {
      return fail;
    }

    // 0.2. Sanity check, skip already fused.
    for (auto inp : laGeneric.inputs()) {
      if (auto fused = inp.template getDefiningOp<miopen::GpuAllocOp>())
        return fail;
    }

    auto idxMaps =
        laGeneric->template getAttrOfType<ArrayAttr>("indexing_maps");
    auto outIdxMap = idxMaps[idxMaps.size() - 1].template cast<AffineMapAttr>();

    if (!outIdxMap.isIdentity()) {
      return fail;
    }

    // 1. Trace input to threadwise_copy. Collect transforms (to be applied to
    // other inputs).
    // 1.1. Find the conv2d output
    Value twinp;
    bool v2 = true;
    SmallVector<Value, 5> transforms;
    for (auto pair : llvm::enumerate(laGeneric.inputs())) {
      int32_t idx = pair.index();
      auto inp = pair.value();
      // 1.1. first trace to back to regs, then forward to twcopy
      SmallVector<Value, 5> transforms_l;
      auto twinp_t =
          traceToThreadwiseCopy<miopen::ThreadwiseCopyV2Op>(inp, transforms_l);
      if (!twinp_t) {
        transforms_l.clear();
        twinp_t =
            traceToThreadwiseCopy<miopen::ThreadwiseCopyOp>(inp, transforms_l);
        if (twinp_t)
          v2 = false;
      }
      if (twinp_t) {
        // 1.2. Only one input should trace to twcopy
        assert(!twinp);
        twinp = twinp_t;
        transforms = transforms_l;
        break;
      } else {
        // 2.1. Test aligned input with output type
        auto inpIdxMap = idxMaps[idx].template cast<AffineMapAttr>();
        if (!checkCompatibleTypes(inp.getType(), inpIdxMap, out.getType(),
                                  outIdxMap)) {
          return fail;
        }
      }
    }

    // 2. Apply if input found
    if (twinp) {
      if (!v2) {
        auto lastTransform = transforms.back();
        auto twcopy = dyn_cast<miopen::ThreadwiseCopyOp>(
            lastTransform.use_begin()->getOwner());

        Value regTWCopy = backtrace<miopen::ThreadwiseCopyOp>(lastTransform, 0);
        if (auto regTransform =
                regTWCopy.getDefiningOp<miopen::TransformOp>()) {
          // 2.0. Reset insertion point to just before threadwise_copy
          b.setInsertionPoint(twcopy);

          // 2.1. Tile and insert linalg.generic on registers
          auto regTFInp = regTransform.input();

          auto laOutRegs =
              reconfigureLAGeneric(laGeneric, b, transforms, regTFInp, twcopy);

          // 2.4. Move linalg.generic
          laGeneric->moveBefore(twcopy);

          // 2.5. Reset input regs on threadwise_copy
          regTransform->setOperand(0, laOutRegs);
          regTransform->moveBefore(twcopy);
          twcopy->setOperand(0, regTransform);

          // 2.6. Reset output on threadwise_copy
          auto mrReshape =
              transforms.front().getDefiningOp<miopen::TransformOp>();
          mrReshape->setOperand(0, out);

          return success();
        }
      } else {
        auto lastTransform = transforms.back();
        SmallVector<Operation *, 2> twcopys;
        for (auto &use : lastTransform.getUses()) {
          if (auto twcopy =
                  dyn_cast<miopen::ThreadwiseCopyV2Op>(use.getOwner())) {
            twcopys.push_back(twcopy);
          } else {
            return fail;
          }
        }
        auto twcopy = dyn_cast<miopen::ThreadwiseCopyV2Op>(twcopys.back());

        Value regBWGemmV2 = twcopy.getOperand(0);
        if (auto miSliceV2 =
                dyn_cast<miopen::ExtractSliceOp>(regBWGemmV2.getDefiningOp())) {
          // 2.0. Reset insertion point to just before threadwise_copy
          b.setInsertionPoint(twcopy);

          // 2.1. Capture gemm return shape
          auto regVecType = regBWGemmV2.getType().template cast<VectorType>();
          assert(regVecType.hasStaticShape());
          assert(regVecType.getRank() == 1);

          SmallVector<int64_t, 2> shape{regVecType.getNumElements()};
          auto elemType = regVecType.getElementType();

          // 2.2. Make vgpr alloc to fit gemm return
          // --- alternatively make 2 linalg.generics (1 for each vector)
          auto regType = MemRefType::get(shape, elemType, {}, 5);
          auto laInRegs = b.create<miopen::GpuAllocOp>(loc, regType);

          // 2.3. Copy gemm result vectors into vgpr
          // > vector.store %58#0, %59[%c0, %c0] : memref<2x4xf32>,
          // vector<4xf32> > vector.store %58#1, %59[%c1, %c0] :
          // memref<2x4xf32>, vector<4xf32>
          Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
          Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
          const SmallVector<Value, 2> coords0{c0, c0};
          const SmallVector<Value, 2> coords1{c1, c0};
          b.create<vector::StoreOp>(loc, twcopys.back()->getOperand(0),
                                    laInRegs, c0);

          // 2.4. Tile linalg.generic with vgpr as input, return output vgprs
          auto laOutRegs =
              reconfigureLAGeneric(laGeneric, b, transforms, laInRegs, twcopy);
          // 2.4.0. and insert before twcopys
          laGeneric->moveBefore(twcopy);

          // 2.5. Replace twcopy inputs with vector from la result vgpr
          auto vload0 =
              b.create<vector::LoadOp>(loc, regVecType, laOutRegs, c0);

          twcopys.back()->setOperand(0, vload0);

          // 2.6. Reset twcopy output to point to old laGeneric output
          auto mrReshape =
              transforms.front().getDefiningOp<miopen::TransformOp>();
          mrReshape->setOperand(0, out);

          return success();
        }
      }
    }

    return fail;
  }
};

//===- Passes
//-------------------------------------------------------------===//
//===- MIOpenLinalgAlignPass - Align Tiling of Linalg Ops
//-----------------===//
//
void MIOpenLinalgAlignPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<MILARewritePattern<linalg::GenericOp>>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::miopen::createMIOpenLinalgAlignPass() {
  return std::make_unique<MIOpenLinalgAlignPass>();
}
