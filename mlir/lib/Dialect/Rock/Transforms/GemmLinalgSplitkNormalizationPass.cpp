//===- GemmLinalgSplitkNormalizationPass.cpp ------------===//
//
// Copyright 2025 Advanced Micro Devices.
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
//
// This pass modifies linalg.generic for split-k fusions. It converts any
// arith.addf/arith.subf gemmOut, other to arith.addf gemmOut,
// other/splitkFactor.
//
//===-----------------------------------------------------===//
#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKGEMMLINALGSPLITKNORMALIZATIONPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-gemm-linalg-splitk-normalization"

using namespace mlir;
using namespace mlir::rock;

namespace {
class RockGemmLinalgSplitkNormalizationPass
    : public rock::impl::RockGemmLinalgSplitkNormalizationPassBase<
          RockGemmLinalgSplitkNormalizationPass> {
  void runOnOperation() override;
};
} // end namespace

static LogicalResult divideAddBySplitkFactor(linalg::GenericOp genericOp,
                                             Value gemmResult,
                                             int64_t splitKFactor,
                                             GemmFeatures features,
                                             IRRewriter &b) {
  SmallVector<std::tuple<Operation *, int>> adds;
  if (failed(checkValidOutputFusion(genericOp, gemmResult, features, adds)))
    return failure();

  for (auto [arithOp, gemmOutIndex] : adds) {
    assert(gemmOutIndex == 0 || gemmOutIndex == 1);
    LLVM_DEBUG(llvm::dbgs() << "Op to modify: " << arithOp << "\n");
    b.setInsertionPoint(arithOp);
    Value gemmOut = arithOp->getOperand(gemmOutIndex);
    Value otherValue =
        (gemmOutIndex == 0) ? arithOp->getOperand(1) : arithOp->getOperand(0);
    auto splitKFactorValue = createConstantFloatOp(
        b, arithOp->getLoc(), otherValue.getType(), otherValue.getType(),
        static_cast<float>(splitKFactor));
    Value otherBySplitk = b.createOrFold<arith::DivFOp>(
        arithOp->getLoc(), otherValue, splitKFactorValue);
    if (isa<arith::AddFOp>(arithOp)) {
      b.replaceOpWithNewOp<arith::AddFOp>(arithOp, gemmOut, otherBySplitk);
    } else if (isa<arith::SubFOp>(arithOp)) {
      if (gemmOutIndex == 0)
        b.replaceOpWithNewOp<arith::SubFOp>(arithOp, gemmOut, otherBySplitk);
      else
        b.replaceOpWithNewOp<arith::SubFOp>(arithOp, otherBySplitk, gemmOut);
    } else {
      return failure();
    }
  }
  return success();
}

static LogicalResult
rewriteLinalgForSplitK(func::FuncOp &func,
                       BufferDependencyAnalysis &bufferDeps) {
  IRRewriter rewriter(func->getContext());
  SmallVector<GemmOp> gemmOps;

  func.walk([&](GemmOp gemmOp) {
    int64_t splitKFactor = gemmOp.getParams()->getSplitKFactor();
    if (splitKFactor > 1) {
      gemmOps.push_back(gemmOp);
    }
  });
  if (gemmOps.size() > 1)
    return failure();

  if (gemmOps.size() == 1) {
    GemmOp gemmOp = gemmOps[0];
    auto gemmResult = gemmOp.getOutArgument()->get();
    int64_t splitKFactor = gemmOp.getParams()->getSplitKFactor();
    GemmFeatures features = rock::getFeatures(gemmOp);

    // save all `linalg::GenericOp` that read from a gemm output
    auto genericOpOperands =
        traceGemmOutputToGenericOps(gemmResult, func, bufferDeps);

    // GEMM result could come from a block argument, so if it fails, we return
    // success()
    if (failed(genericOpOperands))
      return success();

    // check if generic ops are valid fusions
    for (OpOperand *genericOpOperand : genericOpOperands.value()) {
      auto genericOp = cast<linalg::GenericOp>(genericOpOperand->getOwner());
      LLVM_DEBUG(llvm::dbgs()
                 << "Found linalg::GenericOp that reads GEMM output, let's "
                    "modify it if it has addf and/or subf. Op="
                 << genericOp << "\n");
      auto inputAlloc = findMemrefAlloc(genericOpOperand->get());
      if (failed(inputAlloc))
        return failure();

      if (failed(divideAddBySplitkFactor(genericOp, inputAlloc.value(),
                                         splitKFactor, features, rewriter)))
        return failure();
    }
  }

  return success();
}

class ScalesRewritePattern : public OpRewritePattern<GemmOp> {
public:
  using OpRewritePattern<GemmOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GemmOp op, PatternRewriter &rw) const override {
    Location loc = op.getLoc();
    Value scaleA = op.getScaleA();
    Value scaleB = op.getScaleB();
    if (!scaleA || !scaleB)
      return failure(); // scales are optional
    auto scaleAType = dyn_cast<MemRefType>(scaleA.getType());
    auto scaleBType = dyn_cast<MemRefType>(scaleB.getType());
    if (!scaleAType || !scaleBType)
      return op.emitError("scaleA/scaleB must be memref types");
    Type f8e8m0Type = rw.getF8E8M0Type();
    if (scaleAType.getElementType() == f8e8m0Type &&
        scaleBType.getElementType() == f8e8m0Type)
      return failure(); // nothing to do

    auto scaleAShape = scaleAType.getShape();
    auto scaleBShape = scaleBType.getShape();
    SmallVector<Operation *> opsToErase;
    if (scaleAType.getElementType() != f8e8m0Type) {
      FailureOr<memref::AllocOp> scaleAAlloc = findMemrefAlloc(scaleA);
      SmallVector<rock::TransformOp> transforms;
      (void)rock::untransform(scaleA, transforms);
      if (failed(scaleAAlloc))
        return failure();
      BufferDependencyAnalysis bufferDeps(op->getParentOfType<func::FuncOp>());
      std::optional<SmallVector<OpOperand *>> writers =
          bufferDeps.getWriters(scaleAAlloc.value());
      Value newScaleA = scaleA;
      bool reusedConversion = false;
      if (writers && writers->size() == 1) {
        Operation *writerOp = writers->front()->getOwner();
        if (auto genOp = dyn_cast<linalg::GenericOp>(writerOp)) {
          genOp.getRegion().walk([&](arith::ExtFOp extOp) {
            if (reusedConversion)
              return;
            if (genOp.getOutputs()[0].getDefiningOp<memref::AllocOp>() ==
                scaleAAlloc.value()) {
              if (extOp.getIn().getType() == f8e8m0Type) {
                if (auto blockScaleA = dyn_cast<BlockArgument>(extOp.getIn())) {
                  newScaleA = genOp.getInputs()[blockScaleA.getArgNumber()];
                  opsToErase.push_back(genOp);
                  opsToErase.push_back(scaleAAlloc->getOperation());
                  reusedConversion = true;
                }
              }
            }
          });
        }
        if (reusedConversion) {
          // Reconstruct transform chain and substitute.
          SmallVector<Attribute> transformAttrs;
          for (rock::TransformOp trOp : llvm::reverse(transforms)) {
            transformAttrs.push_back(trOp.getTransformAttr());
          }
          ArrayAttr transformsAttr = rw.getArrayAttr(transformAttrs);
          newScaleA = rock::transform(rw, newScaleA, transformsAttr);
          rw.replaceAllUsesWith(scaleA, newScaleA);
        }
        scaleA = newScaleA;
      } else {
        MemRefType newScaleAType = MemRefType::get(scaleAShape, f8e8m0Type);
        memref::AllocOp newScaleAAlloc =
            memref::AllocOp::create(rw, loc, newScaleAType);
        createTypeConversionLaGeneric(rw, loc, scaleA, newScaleAAlloc);
        scaleA = newScaleAAlloc;
      }
    }
    if (scaleBType.getElementType() != f8e8m0Type) {
      FailureOr<memref::AllocOp> scaleBAlloc = findMemrefAlloc(scaleB);
      SmallVector<rock::TransformOp> transforms;
      (void)rock::untransform(scaleB, transforms);
      if (failed(scaleBAlloc))
        return failure(); 
      BufferDependencyAnalysis bufferDeps(op->getParentOfType<func::FuncOp>());
      std::optional<SmallVector<OpOperand *>> writers =
          bufferDeps.getWriters(scaleBAlloc.value());
      Value newScaleB = scaleB;
      bool reusedConversion = false;
      if (writers && writers->size() == 1) {
        Operation *writerOp = writers->front()->getOwner();
        if (auto genOp = dyn_cast<linalg::GenericOp>(writerOp)) {
          genOp.getRegion().walk([&](arith::ExtFOp extOp) {
            if (reusedConversion)
              return;
            if (genOp.getOutputs()[0].getDefiningOp<memref::AllocOp>() ==
                scaleBAlloc.value()) {
              if (extOp.getIn().getType() == f8e8m0Type) {
                if (auto blockScaleB = dyn_cast<BlockArgument>(extOp.getIn())) {
                  newScaleB = genOp.getInputs()[blockScaleB.getArgNumber()];
                  opsToErase.push_back(genOp);
                  opsToErase.push_back(scaleBAlloc->getOperation());
                  reusedConversion = true;
                }
              }
            }
          });
        }
        if (reusedConversion) {
          SmallVector<Attribute> transformAttrs;
          for (rock::TransformOp trOp : llvm::reverse(transforms)) {
            transformAttrs.push_back(trOp.getTransformAttr());
          }
          ArrayAttr transformsAttr = rw.getArrayAttr(transformAttrs);
          newScaleB = rock::transform(rw, newScaleB, transformsAttr);
          rw.replaceAllUsesWith(scaleB, newScaleB);
        }
        scaleB = newScaleB;
      } else {
        MemRefType newScaleBType = MemRefType::get(scaleBShape, f8e8m0Type);
        memref::AllocOp newScaleBAlloc =
            memref::AllocOp::create(rw, loc, newScaleBType);
        createTypeConversionLaGeneric(rw, loc, scaleB, newScaleBAlloc);
        scaleB = newScaleBAlloc;
      }
    }
    auto newGemm = rw.replaceOpWithNewOp<rock::GemmOp>(
        op, op->getResultTypes(), op.getA(), op.getB(), op.getC(), scaleA,
        scaleB, op.getATransposedAttr(), op.getBTransposedAttr(),
        op.getCTransposedAttr(), op.getAScaleTransposedAttr(),
        op.getBScaleTransposedAttr(), op.getFeaturesAttr(),
        op.getStoreMethodAttr(), op.getDerivedBlockSizeAttr(),
        op.getGridSizeAttr(),
        op.getParams() ? op.getParams().value() : nullptr);
    for (Operation *eraseOp : opsToErase)
      if (eraseOp && eraseOp->use_empty())
        rw.eraseOp(eraseOp);
    (void)newGemm;
    return success();
  }
};

void RockGemmLinalgSplitkNormalizationPass::runOnOperation() {
  func::FuncOp func = getOperation();
  BufferDependencyAnalysis &bufferDeps =
      getAnalysis<BufferDependencyAnalysis>();

  if (failed(rewriteLinalgForSplitK(func, bufferDeps))) {
    return signalPassFailure();
  }
  RewritePatternSet patterns(&getContext());
  patterns.add<ScalesRewritePattern>(&getContext());
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    return signalPassFailure();
  }
} // namespace
