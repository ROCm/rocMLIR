//===- LowerMIOpenOps.cpp - MLIR MIOpen ops lowering passes ---------------===//
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
// This pass converts miopen.conv2d into miopen.transform and
// miopen.gridwise_gemm.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/MIOpen/LowerMIOpenOps.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
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

#include <iostream>

using namespace mlir;

namespace {
struct MIOpenLinalgAlignPass : public MIOpenLinalgAlignPassBase<MIOpenLinalgAlignPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

//===- MILARewritePattern -------------------------------------------------===//
//===-  ------------------------------------------------===//
template <typename T>
struct MILARewritePattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  template <typename Top>
  Value getOpResult(OpOperand &use) const {
    Value v;
    auto *ownr = use.getOwner();
    if (auto op = dyn_cast<Top>(ownr)) {
      v = op->getResult(0);
    }
    return v;
  }

  template <typename Top>
  Value backtrace(Value inv) const {
    if (inv.hasOneUse()) {
      return getOpResult<Top>(*inv.use_begin());
    }
    return Value();
  }

  template <typename Top>
  Value getOpOperand(OpOperand &use, int idx) const {
    Value v;
    auto *ownr = use.getOwner();
    if (auto op = dyn_cast<Top>(ownr)) {
      v = op->getOperand(idx);
    }
    return v;
  }

  template <typename Top>
  Value backtrace(Value inv, int idx) const {
    if (inv.hasOneUse()) {
      return getOpOperand<Top>(*inv.use_begin(), idx);
    }
    return Value();
  }

  template <typename Top>
  Top backtraceOp(Value inv) const {
    if (inv.hasOneUse()) {
      auto *ownr = inv.use_begin()->getOwner();
      return dyn_cast<Top>(ownr);
    }
    return Top();
  }

  Value makeSubview(PatternRewriter &b, miopen::ThreadwiseCopyOp &twcopy, Value inp) const {
    Value subview;

    auto ctx = b.getContext();
    auto loc = inp.getLoc();

    auto outputType = inp.getType().template cast<MemRefType>();
    auto outputShape = outputType.getShape();
    auto outputDims = outputShape.size();
    auto outputElementType = outputType.getElementType();

    // 0. initialize subview params
    Value zero = b.create<ConstantIndexOp>(loc, 0);
    auto zeroA = b.getIndexAttr(0);
    SmallVector<OpFoldResult, 5> offsets(outputDims, zero);
    SmallVector<OpFoldResult, 5> sizes(outputDims, zeroA);
    SmallVector<OpFoldResult, 5> strides(outputDims, zeroA);
    
    // 1. flatten input affine map with miopen::transform, (subview only allows 1 affine map)
    auto expr = getAffineConstantExpr(0, ctx);
    unsigned stride = 1;
    for (int i = outputDims - 1; i >= 0; --i) {
      expr = expr + getAffineDimExpr(i, ctx) *
                 getAffineConstantExpr(stride, ctx);
      strides[i] = b.getIndexAttr(stride);
      stride *= outputShape[i];
    }
    AffineMap outputAffineMap = AffineMap::get(outputDims, 0, ArrayRef<AffineExpr>{expr}, ctx);
    auto transformedOutputType = MemRefType::get(outputShape, outputElementType, {outputAffineMap});
    
    llvm::SmallVector<NamedAttribute, 3> transformedNewOutputAttrs;
    auto transform = b.create<miopen::TransformOp>(loc, transformedOutputType, inp, transformedNewOutputAttrs, true);

    // 2. reduce scope of inp to tile size (use subview: https://mlir.llvm.org/docs/Dialects/MemRef/#memrefsubview-mlirmemrefsubviewop Ex 3)
    auto regs = twcopy.getOperand(0);
    auto regShape = regs.getType().template cast<MemRefType>().getShape();
    auto regDims = regShape.size();
    for (uint32_t i = 0; i < regDims; ++i) {
      Value idx = b.create<IndexCastOp>(loc, twcopy.getOperand(2 + regDims + i), b.getIndexType());
      offsets[i] = idx;
      sizes[i] = b.getIndexAttr(regShape[i]);
    }

    return b.create<SubViewOp>(loc, transform, offsets, sizes, strides);
  }

  Value traceToThreadwiseCopy(Value inp, SmallVector<Value, 5> &transforms) const {
    Value ret;
    Value laReshape;
    // get reader (linagl.reshape), return result
    int cnt = 0;
    for (auto &use : inp.getUses()) {
      if (auto op = dyn_cast<linalg::GenericOp>(use.getOwner())) {
        // reader
      } else if (!laReshape) {
        laReshape = getOpResult<linalg::ReshapeOp>(use);
      }
      cnt++;
    }
    if (cnt > 2) {
      return ret;
    }
    if (laReshape) {
      transforms.push_back(laReshape);
      // get reader (miopen.transform), return result
      Value miTransform = backtrace<miopen::TransformOp>(laReshape);
      transforms.push_back(miTransform);
      Value miTransform2 = backtrace<miopen::TransformOp>(miTransform);
      transforms.push_back(miTransform2);
      auto twcopy = dyn_cast<miopen::ThreadwiseCopyOp>(miTransform2.use_begin()->getOwner());
      if (twcopy)
        ret = inp;
    }
    return ret;
  }
  
  Value applyTransforms(PatternRewriter &b, miopen::ThreadwiseCopyOp &twcopy, Value inp, SmallVector<Value, 5> &transforms) const {
    Value ret = inp;
    BlockAndValueMapping cloningMap;
    for (auto transform : transforms) {
      assert(transform.hasOneUse());
      Operation *tcopy;
      if (auto miTransform = transform.getDefiningOp<miopen::TransformOp>()) {
        cloningMap.map(miTransform->getOperand(0), ret);
        tcopy = b.clone(*miTransform, cloningMap);
      } else if (auto laReshape = transform.getDefiningOp<linalg::ReshapeOp>()) {
        cloningMap.map(laReshape->getOperand(0), ret);
        tcopy = b.clone(*laReshape, cloningMap);
      } else {
        assert(0);
      }
      ret = tcopy->getResult(0);
    }

    // create sub-view based on threadwise_copy
    return makeSubview(b, twcopy, ret);
  }
  
  LogicalResult matchAndRewrite(T op, PatternRewriter &b) const override {
    LogicalResult res = failure();
    auto loc = op.getLoc();
    auto ctx = op.getContext();

    // 0. Test compatibility
    // 0.0. Only fully parallel for now
    for (auto itr : op.iterator_types()) {
      if (itr.template cast<StringAttr>().getValue() != "parallel") {
        return res;
      }
    }

    Value out = *op.outputs().begin(); // may be another arg
    // 0.1. Test compatibility,  Only 1 output supported
    if (op.outputs().size() > 1) {
      return res;
    }

    Value twinp;
    SmallVector<Value, 5> transforms;
    // 1. Trace input to threadwise_copy. Collect transforms (to be applied to other inputs). test compatibility
    for (auto inp : op.inputs()) {
      // 1.1. Test aligned input with output type
      if (inp.getType() != out.getType()) {
        return res;
      }
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
      auto twcopy = dyn_cast<miopen::ThreadwiseCopyOp>(lastTransform.use_begin()->getOwner());

      // 2.0. Reset insertion point to just before threadwise_copy
      b.setInsertionPoint(twcopy);

      Value regTWCopy = backtrace<miopen::ThreadwiseCopyOp>(lastTransform, 0);
      if (auto regTransform = regTWCopy.getDefiningOp<miopen::TransformOp>()) {
        // 2.1. Tile and insert linalg.generic on registers
        auto regType = regTransform.getType();
        auto oRegs = b.create<miopen::GpuAllocOp>(loc, regType);

        SmallVector<AffineMap, 5> laGenericAMaps;
        SmallVector<Value, 5> newInputs;
        for (auto inp : op.inputs()) {
          Value newInput;
          if (inp == twinp) {
            newInput = regTransform;
          } else {
            // 2.1.1. Align tiling of other inputs
            newInput = applyTransforms(b, twcopy, inp, transforms);
          }
          newInputs.push_back(newInput);
          laGenericAMaps.push_back(AffineMap::getMultiDimIdentityMap(newInput.getType().template cast<MemRefType>().getRank(), ctx));
        }
        laGenericAMaps.push_back(AffineMap::getMultiDimIdentityMap(regType.getRank(), ctx));

        op.inputsMutable().assign(newInputs);
        op.outputsMutable().assign(oRegs);

        // 2.2. Reset affine maps
        op.indexing_mapsAttr(b.getAffineMapArrayAttr(laGenericAMaps));

        // 2.3. Reset iterator types
        SmallVector<StringAttr, 5> laGenericIteratorArr(regType.getRank(), b.getStringAttr("parallel"));
        op.iterator_typesAttr(b.getArrayAttr(ArrayRef<Attribute>(laGenericIteratorArr.begin(), laGenericIteratorArr.end())));

        // 2.4. Move linalg.generic
        op->moveBefore(twcopy);

        // 2.5. Reset input regs on threadwise_copy
        twcopy->setOperand(0, oRegs);

        // 2.6. Reset output on threadwise_copy
        auto laReshape = transforms.front().getDefiningOp<linalg::ReshapeOp>();
        auto outOp = out.getDefiningOp<AllocOp>();
        outOp->moveBefore(laReshape);
        laReshape->setOperand(0, out);

        res = success();
      }
    }
    
    return res;
  }
};

//===- Passes -------------------------------------------------------------===//
//===- MIOpenLinalgAlignPass - Align Tiling of Linalg Ops -----------------===//
//
void MIOpenLinalgAlignPass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<MILARewritePattern<linalg::GenericOp>>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}


std::unique_ptr<Pass> mlir::miopen::createMIOpenLinalgAlignPass() {
  return std::make_unique<MIOpenLinalgAlignPass>();
}

