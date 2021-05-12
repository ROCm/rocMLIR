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

using namespace mlir;

namespace {
struct LowerMIOpenOpsStep1Pass : public MIOpenOpsStep1PassBase<LowerMIOpenOpsStep1Pass> {
  void runOnOperation() override;
};

struct LowerMIOpenOpsStep2Pass : public MIOpenOpsStep2PassBase<LowerMIOpenOpsStep2Pass> {
  void runOnOperation() override;
};

struct LowerMIOpenOpsStep3Pass : public MIOpenOpsStep3PassBase<LowerMIOpenOpsStep3Pass> {
  void runOnOperation() override;
};

struct LowerMIOpenOpsStep4Pass
    : public MIOpenOpsStep4PassBase<LowerMIOpenOpsStep4Pass> {
  void runOnOperation() override;
};

struct LowerMIOpenOpsStep5Pass
    : public MIOpenOpsStep5PassBase<LowerMIOpenOpsStep5Pass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// High level convolution operation always have
// [filter, input, output]
// as the convolution argument. The only difference between different
// hight level convolution operations is the argument sequence. For
// simplicity, we always arrange the first two arguments to be input
// and the last argument to be output

template <>
const ArgumentFields Conv2DRewritePattern<miopen::Conv2DOp>::fields = {
    {0, 1, 2},
    {"KM", "KN", "MN"},
};
template <>
const miopen::ConvOpType Conv2DRewritePattern<miopen::Conv2DOp>::convOpType =
    miopen::ConvOpType::Conv2DOpType;

template <>
const ArgumentFields Conv2DRewritePattern<miopen::Conv2DBwdDataOp>::fields = {
    {0, 2, 1},
    {"KM", "MN", "KN"},
};

template <>
const miopen::ConvOpType
    Conv2DRewritePattern<miopen::Conv2DBwdDataOp>::convOpType =
        miopen::ConvOpType::Conv2DBwdDataOpType;

template <>
const ArgumentFields Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>::fields = {
    {2, 1, 0},
    {"MN", "KN", "KM"},
};

template <>
const miopen::ConvOpType
    Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>::convOpType =
        miopen::ConvOpType::Conv2DBwdWeightOpType;

// Explicitly instantiate the template to operation type
template struct Conv2DRewritePattern<miopen::Conv2DOp>;
template struct Conv2DRewritePattern<miopen::Conv2DBwdDataOp>;
template struct Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>;

void LowerMIOpenOpsStep1Pass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DOp>>(&getContext());
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DBwdDataOp>>(&getContext());
  patterns.insert<Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>>(
      &getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void LowerMIOpenOpsStep2Pass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<GridwiseGemmRewritePattern>(&getContext());
  patterns.insert<GridwiseGemmV2RewritePattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void LowerMIOpenOpsStep3Pass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<FillRewritePattern>(&getContext());
  patterns.insert<MovePosV2RewritePattern>(&getContext());
  patterns.insert<SubviewRewritePattern>(&getContext());
  patterns.insert<TransformRewritePattern>(&getContext());
  patterns.insert<BlockwiseGemmRewritePattern>(&getContext());
  patterns.insert<BlockwiseGemmV2RewritePattern>(&getContext());
  patterns.insert<BlockwiseCopyRewritePattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void LowerMIOpenOpsStep4Pass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<ThreadwiseGemmRewritePattern>(&getContext());
  patterns.insert<ThreadwiseCopyRewritePattern>(&getContext());
  patterns.insert<ThreadwiseCopyV2RewritePattern>(&getContext());
  patterns.insert<XdlopsGemmV2RewritePattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void LowerMIOpenOpsStep5Pass::runOnOperation() {
  OwningRewritePatternList patterns;
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep1Pass() {
  return std::make_unique<LowerMIOpenOpsStep1Pass>();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep2Pass() {
  return std::make_unique<LowerMIOpenOpsStep2Pass>();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep3Pass() {
  return std::make_unique<LowerMIOpenOpsStep3Pass>();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep4Pass() {
  return std::make_unique<LowerMIOpenOpsStep4Pass>();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep5Pass() {
  return std::make_unique<LowerMIOpenOpsStep5Pass>();
}
