//===- CopyOpt.cpp - Remove redundant memories
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
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
struct MIOpenCopyOptPass : public MIOpenCopyOptPassBase<MIOpenCopyOptPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

//===- MICORewritePattern -------------------------------------------------===//
//===-  ------------------------------------------------===//
template <typename T> struct MICORewritePattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op, PatternRewriter &b) const override {
    LogicalResult fail = failure();
    auto loc = op.getLoc();
    auto ctx = op.getContext();

    // 0. Test compatibility
    // 0.0 Global Memory Space
    auto allocType = op.getType().template cast<MemRefType>();
    auto memSpace = allocType.getMemorySpaceAsInt();
    if (memSpace == 3 || memSpace == 5)
      return fail;

    Value mem = op->getResult(0);

    // 1. Capture allocation->copy pattern
    Operation *writer = nullptr;
    memref::CopyOp reader = nullptr;
    for (auto &use : mem.getUses()) {
      if (auto laop = dyn_cast<linalg::GenericOp>(use.getOwner())) {
        // 0.1 Output of linalg.generic
        if (writer)
          return fail;
        for (auto out : laop.outputs()) {
          if (out == mem)
            writer = laop;
        }
        if (!writer)
          return fail;
      } else if (auto mrop = dyn_cast<memref::CopyOp>(use.getOwner())) {
        // 0.2 Only one final memref.copy into interface memref
        if (reader)
          return fail;
        if (mrop.getSource() != mem)
          return fail;
        reader = mrop;
      } else {
        // unsupported op
        return fail;
      }
    }

    // do it
    if (reader && writer) {
      auto realMem = reader.getTarget();

      mem.replaceAllUsesWith(realMem);

      reader->erase();

      return success();
    }

    return fail;
  }
};

//===- Passes -------------------------------------------------------------===//
//===- MIOpenCopyOptPass -  -----------------===//
//
void MIOpenCopyOptPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  OwningRewritePatternList patterns(ctx);
  patterns.insert<MICORewritePattern<memref::AllocOp>>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::miopen::createMIOpenCopyOptPass() {
  return std::make_unique<MIOpenCopyOptPass>();
}
