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

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

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

  Value backtraceToAlloc(Value inp) const {
    if (auto bt = inp.getDefiningOp<miopen::TransformOp>())
      return backtraceToAlloc(bt->getOperand(0));
    if (auto bt = inp.getDefiningOp<memref::ExpandShapeOp>())
      return backtraceToAlloc(bt->getOperand(0));
    if (auto bt = inp.getDefiningOp<memref::AllocOp>())
      return inp;
    if (auto bt = inp.getDefiningOp<miopen::GpuAllocOp>())
      return inp;
    return Value();
  }

  Value makeThreadwiseCopy(PatternRewriter &b, Operation *miTWCopy,
                           Value inp) const {
    // 0. capture reg alloc for miTWCopy output regs
    auto twinp = miTWCopy->getOperand(0);
    auto miTransform = twinp.getDefiningOp<miopen::TransformOp>();
    auto miTransformInp = miTransform->getOperand(0);
    auto miAlloc = miTransformInp.getDefiningOp<miopen::GpuAllocOp>();

    // 1. clone reg alloc + transform
    BlockAndValueMapping cloningMap;
    auto nAlloc = b.clone(*miAlloc, cloningMap);
    cloningMap.map(miTransform->getOperand(0), nAlloc->getResult(0));
    auto nTransform = b.clone(*miTransform, cloningMap);

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

    // 5. Adjust the copy to show the correct argument as global
    nTWCopy->setAttr("globalArg", b.getIndexAttr(0));

    return nAlloc->getResult(0);
  }

  Value applyTransforms(PatternRewriter &b, Operation *miTWCopy, Value inp,
                        SmallVector<Value, 5> &transforms) const {
    Value ret = inp;
    BlockAndValueMapping cloningMap;
    // 1. clone the same transforms applied to the output memory and
    //    apply to all other inputs to the linalg.generic
    for (auto transform : transforms) {
      assert(transform.hasOneUse());
      Operation *tcopy;
      if (auto miTransform = transform.getDefiningOp<miopen::TransformOp>()) {
        cloningMap.map(miTransform->getOperand(0), ret);
        tcopy = b.clone(*miTransform, cloningMap);
      } else if (auto laReshape =
                     transform.getDefiningOp<memref::ExpandShapeOp>()) {
        cloningMap.map(laReshape->getOperand(0), ret);
        tcopy = b.clone(*laReshape, cloningMap);
      } else {
        assert(0);
      }
      ret = tcopy->getResult(0);
    }

    // 2. also create threadwise_copy from global to regs
    //    TODO(sjw): make sure output buffer writes (means these inputs will be
    //    buffer reads)
    return makeThreadwiseCopy(b, miTWCopy, ret);
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
        nextVal = getOpResult<memref::ExpandShapeOp>(use);
        transforms.push_back(nextVal);
      }
      cnt++;
    }
    if (cnt > 2)
      return null;

    while (nextVal) {
      // 2. get reader (miopen.transform), return result
      if (auto transform = backtrace<miopen::TransformOp>(nextVal)) {
        nextVal = transform;
        transforms.push_back(nextVal);
      } else {
        // 3. get readers (miopen.threadwise_copy)
        for (auto &use : nextVal.getUses()) {
          if (!dyn_cast<Ttwcopy>(use.getOwner())) {
            return null;
          }
        }
        return inp;
      }
    }
    return null;
  }

  Value reconfigureLAGeneric(T &laGeneric, PatternRewriter &b,
                             SmallVector<Value, 5> &transforms, Value laIn,
                             Operation *twcopy) const {
    auto ctx = laGeneric.getContext();
    auto loc = laGeneric.getLoc();
    Value twout = backtraceToAlloc(twcopy->getOperand(1));
    auto regType = laIn.getType().template cast<MemRefType>();
    auto laOut = b.create<miopen::GpuAllocOp>(loc, regType);

    SmallVector<AffineMap, 5> laGenericAMaps;
    SmallVector<Value, 5> newInputs;
    for (auto inp : laGeneric.inputs()) {
      Value newInput;
      if (inp == twout) {
        newInput = laIn;
      } else {
        // 2.1.1. Align tiling of other inputs
        newInput = applyTransforms(b, twcopy, inp, transforms);
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
    SmallVector<StringAttr, 5> laGenericIteratorArr(
        regType.getRank(), b.getStringAttr("parallel"));
    laGeneric.iterator_typesAttr(b.getArrayAttr(ArrayRef<Attribute>(
        laGenericIteratorArr.begin(), laGenericIteratorArr.end())));
    return laOut;
  }

  LogicalResult matchAndRewrite(T laGeneric,
                                PatternRewriter &b) const override {
    LogicalResult fail = failure();
    auto loc = laGeneric.getLoc();

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

    Value twinpV1;
    Value twinpV2;
    SmallVector<Value, 5> transforms;
    // 1. Trace input to threadwise_copy. Collect transforms (to be applied to
    // other inputs). test compatibility
    for (auto inp : laGeneric.inputs()) {
      // 1.1. Test aligned input with output type
      if (inp.getType() != out.getType()) {
        return fail;
      }
      // first trace to back to regs, then forward to twcopy
      if (auto twinp_t = traceToThreadwiseCopy<miopen::ThreadwiseCopyOp>(
              inp, transforms)) {
        // 1.2. Only one input should trace to twcopy
        assert(!twinpV1);
        twinpV1 = twinp_t;
      } else if (auto twinp_t =
                     traceToThreadwiseCopy<miopen::ThreadwiseCopyV2Op>(
                         inp, transforms)) {
        assert(!twinpV2);
        twinpV2 = twinp_t;
      }
    }

    // 2. Apply if input found
    if (twinpV1) {
      auto lastTransform = transforms.back();
      auto twcopy = dyn_cast<miopen::ThreadwiseCopyOp>(
          lastTransform.use_begin()->getOwner());

      Value regTWCopy = backtrace<miopen::ThreadwiseCopyOp>(lastTransform, 0);
      if (auto regTransform = regTWCopy.getDefiningOp<miopen::TransformOp>()) {
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
            transforms.front().getDefiningOp<memref::ExpandShapeOp>();
        mrReshape->setOperand(0, out);

        return success();
      }
    } else if (twinpV2) {
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
      if (twcopys.size() != 2)
        return fail;

      auto twcopy = dyn_cast<miopen::ThreadwiseCopyV2Op>(twcopys.back());

      Value regBWGemmV2 = twcopy.getOperand(0);
      if (auto miBWGemmV2 =
              regBWGemmV2.getDefiningOp<miopen::BlockwiseGemmV2Op>()) {
        // 2.0. Reset insertion point to just before threadwise_copy
        b.setInsertionPoint(twcopy);

        // 2.1. Capture gemm return shape
        auto regVecType = regBWGemmV2.getType().template cast<VectorType>();
        assert(regVecType.hasStaticShape());
        assert(regVecType.getRank() == 1);

        SmallVector<int64_t, 2> shape{2, regVecType.getNumElements()};
        auto elemType = regVecType.getElementType();

        // 2.2. Make vgpr alloc to fit gemm return
        // --- alternatively make 2 linalg.generics (1 for each vector)
        auto regType = MemRefType::get(shape, elemType, {}, 5);
        auto laInRegs = b.create<miopen::GpuAllocOp>(loc, regType);

        // 2.3. Copy gemm result vectors into vgpr
        // > vector.store %58#0, %59[%c0, %c0] : memref<2x4xf32>, vector<4xf32>
        // > vector.store %58#1, %59[%c1, %c0] : memref<2x4xf32>, vector<4xf32>
        Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
        const SmallVector<Value, 2> coords0{c0, c0};
        const SmallVector<Value, 2> coords1{c1, c0};
        b.create<vector::StoreOp>(loc, twcopys.back()->getOperand(0), laInRegs,
                                  coords0);
        b.create<vector::StoreOp>(loc, twcopys.front()->getOperand(0), laInRegs,
                                  coords1);

        // 2.4. Tile linalg.generic with vgpr as input, return output vgprs
        auto laOutRegs =
            reconfigureLAGeneric(laGeneric, b, transforms, laInRegs, twcopy);
        // 2.4.0. and insert before twcopys
        laGeneric->moveBefore(twcopy);

        // 2.5. Replace twcopy inputs with vector from la result vgpr
        auto vload0 =
            b.create<vector::LoadOp>(loc, regVecType, laOutRegs, coords0);
        auto vload1 =
            b.create<vector::LoadOp>(loc, regVecType, laOutRegs, coords1);
        twcopys.back()->setOperand(0, vload0);
        twcopys.front()->setOperand(0, vload1);

        // 2.6. Reset twcopy output to point to old laGeneric output
        auto mrReshape =
            transforms.front().getDefiningOp<memref::ExpandShapeOp>();
        mrReshape->setOperand(0, out);
      }
    }

    return fail;
  }
};

//===- Passes -------------------------------------------------------------===//
//===- MIOpenLinalgAlignPass - Align Tiling of Linalg Ops -----------------===//
//
void MIOpenLinalgAlignPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  OwningRewritePatternList patterns(ctx);
  patterns.insert<MILARewritePattern<linalg::GenericOp>>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::miopen::createMIOpenLinalgAlignPass() {
  return std::make_unique<MIOpenLinalgAlignPass>();
}
