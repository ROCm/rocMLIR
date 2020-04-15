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

#include "mlir/Dialect/MIOpenOps/LowerMIOpenOps.h"

#include "mlir/Dialect/MIOpenOps/MIOpenOps.h"
#include "mlir/Dialect/MIOpenOps/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {
struct LowerMIOpenOpsStep1Pass : public ModulePass<LowerMIOpenOpsStep1Pass> {
  void runOnModule() override;
};

struct LowerMIOpenOpsStep2Pass : public ModulePass<LowerMIOpenOpsStep1Pass> {
  void runOnModule() override;
};

struct LowerMIOpenOpsStep3Pass : public ModulePass<LowerMIOpenOpsStep1Pass> {
  void runOnModule() override;
};
} // end anonymous namespace

void LowerMIOpenOpsStep1Pass::runOnModule() {
  OwningRewritePatternList patterns;
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DOp>>(&getContext());
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DBwdDataOp>>(&getContext());
  applyPatternsGreedily(getModule(), patterns);
}

void LowerMIOpenOpsStep2Pass::runOnModule() {
  OwningRewritePatternList patterns;
  patterns.insert<GridwiseGemmRewritePattern>(&getContext());
  applyPatternsGreedily(getModule(), patterns);
}

void LowerMIOpenOpsStep3Pass::runOnModule() {
  OwningRewritePatternList patterns;
  patterns.insert<BlockwiseGemmRewritePattern>(&getContext());
  patterns.insert<BlockwiseCopyRewritePattern>(&getContext());
  applyPatternsGreedily(getModule(), patterns);
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::miopen::createLowerMIOpenOpsStep1Pass() {
  return std::make_unique<LowerMIOpenOpsStep1Pass>();
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::miopen::createLowerMIOpenOpsStep2Pass() {
  return std::make_unique<LowerMIOpenOpsStep2Pass>();
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::miopen::createLowerMIOpenOpsStep3Pass() {
  return std::make_unique<LowerMIOpenOpsStep3Pass>();
}

static PassRegistration<LowerMIOpenOpsStep1Pass>
    lowerMIOpenOpsStep1Pass("miopen-lowering",
                       "Lower MIOpen conv2d into transform and gridwise_gemm.");

static PassRegistration<LowerMIOpenOpsStep2Pass>
    lowerMIOpenOpsStep2Pass("miopen-lowering-step2",
                       "Lower MIOpen gridwise_gemm into blockwise ops.");

static PassRegistration<LowerMIOpenOpsStep3Pass>
    lowerMIOpenOpsStep3Pass("miopen-lowering-step3",
                       "Lower MIOpen blockwise ops into threadwise ops.");
