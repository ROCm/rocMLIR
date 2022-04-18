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
// This pass removes redundant local memories.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

    // 0. Test compatibility
    // 0.0 Global Memory Space
    auto allocType = op.getType().template cast<MemRefType>();
    auto memSpace = allocType.getMemorySpaceAsInt();
    if (memSpace == gpu::GPUDialect::getWorkgroupAddressSpace() ||
        memSpace == gpu::GPUDialect::getPrivateAddressSpace())
      return fail;

    Value mem = op->getResult(0);

    // 1. Capture allocation->copy pattern
    Operation *writer = nullptr;
    memref::CopyOp reader = nullptr;
    for (auto &use : mem.getUses()) {
      if (auto laop = dyn_cast<linalg::GenericOp>(use.getOwner())) {
        // 1.0 Output of linalg.generic
        if (writer)
          return fail;
        for (auto out : laop.outputs()) {
          if (out == mem)
            writer = laop;
        }
        if (!writer)
          return fail;
      } else if (auto mrop = dyn_cast<miopen::TransformOp>(use.getOwner())) {
        // 1.1 Input of miopen.transform
        if (writer)
          return fail;
        Value mrval = mrop;
        // 1.1.0 Confirm output of miopen.conv2d
        int cnt = 0;
        for (auto &mruse : mrval.getUses()) {
          if (auto convop = dyn_cast<miopen::Conv2DOp>(mruse.getOwner())) {
            if (convop.getOperand(2) != mrval)
              return fail;
          }
          cnt++;
        }
        if (cnt != 1)
          return fail;
        writer = mrop;
      } else if (auto callop = dyn_cast<CallOpInterface>(use.getOwner())) {
        // 1.3 Assume call is the writer (fails for multiple calls)
        if (writer)
          return fail;
        writer = callop;
      } else if (auto mrop = dyn_cast<memref::CopyOp>(use.getOwner())) {
        // 1.2 Only one final memref.copy into interface memref
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

    // 2. do it
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
  RewritePatternSet patterns(ctx);
  patterns.add<MICORewritePattern<memref::AllocOp>>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::miopen::createMIOpenCopyOptPass() {
  return std::make_unique<MIOpenCopyOptPass>();
}
