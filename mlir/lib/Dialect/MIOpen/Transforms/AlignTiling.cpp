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
#include "mlir/Dialect/MIOpen/LowerMIOpenOps.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

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

  Value makeThreadwiseCopy(PatternRewriter &b, miopen::ThreadwiseCopyOp &miTWCopy,
                           Value inp) const {
    // 0. capture reg alloc for miTWCopy output regs
    auto twinp = miTWCopy.getOperand(0);
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
    for (uint i=0; i<5; ++i) {
      auto inCoord = nTWCopy->getOperand(2+i);
      auto outCoord = nTWCopy->getOperand(2+5+i);
      nTWCopy->setOperand(2+i, outCoord);
      nTWCopy->setOperand(2+5+i, inCoord);
    }

    return nAlloc->getResult(0);
  }

  Value applyTransforms(PatternRewriter &b, miopen::ThreadwiseCopyOp &miTWCopy,
                        Value inp, SmallVector<Value, 5> &transforms) const {
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
    //    TODO(sjw): make sure output buffer writes (means these inputs will be buffer reads)
    return makeThreadwiseCopy(b, miTWCopy, ret);
  }

  Value traceToThreadwiseCopy(Value inp,
                              SmallVector<Value, 5> &transforms) const {
    Value ret;
    Value mrReshape;
    // 1. get reader (memref.expand_shape), return result
    int cnt = 0;
    for (auto &use : inp.getUses()) {
      if (auto op = dyn_cast<linalg::GenericOp>(use.getOwner())) {
        // reader
      } else if (!mrReshape) {
        mrReshape = getOpResult<memref::ExpandShapeOp>(use);
      }
      cnt++;
    }
    if (cnt > 2) {
      return ret;
    }
    if (mrReshape) {
      transforms.push_back(mrReshape);
      // 2. get reader (miopen.transform), return result
      Value miTransform = backtrace<miopen::TransformOp>(mrReshape);
      transforms.push_back(miTransform);
      Value miTransform2 = backtrace<miopen::TransformOp>(miTransform);
      transforms.push_back(miTransform2);
      auto twcopy = dyn_cast<miopen::ThreadwiseCopyOp>(
          miTransform2.use_begin()->getOwner());
      if (twcopy)
        ret = inp;
    }
    return ret;
  }

  LogicalResult matchAndRewrite(T laGeneric, PatternRewriter &b) const override {
    LogicalResult fail = failure();
    auto loc = laGeneric.getLoc();
    auto ctx = laGeneric.getContext();

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

    Value twinp;
    SmallVector<Value, 5> transforms;
    // 1. Trace input to threadwise_copy. Collect transforms (to be applied to
    // other inputs). test compatibility
    for (auto inp : laGeneric.inputs()) {
      // 1.1. Test aligned input with output type
      if (inp.getType() != out.getType()) {
        return fail;
      }
      // first trace to back to regs, then forward to twcopy
      auto twinp_t = traceToThreadwiseCopy(inp, transforms);
      
      if (twinp_t) {
        // 1.2. Only one input should trace to twcopy
        assert(!twinp);
        twinp = twinp_t;
      }
    }

    // 2. Apply if input found
    if (twinp) {
      auto lastTransform = transforms.back();
      auto twcopy = dyn_cast<miopen::ThreadwiseCopyOp>(
          lastTransform.use_begin()->getOwner());

      // 2.0. Reset insertion point to just before threadwise_copy
      b.setInsertionPoint(twcopy);

      Value regTWCopy = backtrace<miopen::ThreadwiseCopyOp>(lastTransform, 0);
      if (auto regTransform = regTWCopy.getDefiningOp<miopen::TransformOp>()) {
        // 2.1. Tile and insert linalg.generic on registers
        auto regTFInp = regTransform.input();
        auto regType = regTFInp.getType().template cast<MemRefType>();
        auto oRegs = b.create<miopen::GpuAllocOp>(loc, regType);

        SmallVector<AffineMap, 5> laGenericAMaps;
        SmallVector<Value, 5> newInputs;
        for (auto inp : laGeneric.inputs()) {
          Value newInput;
          if (inp == twinp) {
            newInput = regTFInp;
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
        laGeneric.outputsMutable().assign(oRegs);

        // 2.2. Reset affine maps
        laGeneric.indexing_mapsAttr(b.getAffineMapArrayAttr(laGenericAMaps));

        // 2.3. Reset iterator types
        SmallVector<StringAttr, 5> laGenericIteratorArr(
            regType.getRank(), b.getStringAttr("parallel"));
        laGeneric.iterator_typesAttr(b.getArrayAttr(ArrayRef<Attribute>(
            laGenericIteratorArr.begin(), laGenericIteratorArr.end())));

        // 2.4. Move linalg.generic
        laGeneric->moveBefore(twcopy);

        // 2.5. Reset input regs on threadwise_copy
        regTransform->setOperand(0, oRegs);
        regTransform->moveBefore(twcopy);
        twcopy->setOperand(0, regTransform);

        // 2.6. Reset output on threadwise_copy
        auto mrReshape = transforms.front().getDefiningOp<memref::ExpandShapeOp>();
        mrReshape->setOperand(0, out);

        return success();
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
