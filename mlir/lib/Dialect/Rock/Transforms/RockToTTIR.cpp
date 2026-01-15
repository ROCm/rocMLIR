//===- RockToTTIR - MLIR Rock ops lowering passes -------------------------===//
//
// Copyright 2026 The MLIR Authors.
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
// This pass converts Rock dialect operations to Triton IR
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
// #include "triton/Dialect/Triton/IR/TritonEnums.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKTOTTIRPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-to-ttir"

using namespace mlir;
using namespace mlir::rock;
using namespace mlir::triton;

namespace {
struct RockToTTIRPass
    : public rock::impl::RockToTTIRPassBase<RockToTTIRPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// RockArithOpRewritePattern - Convert rock.arith_op to tt.get_program_id
// for testing TTIR generation
//===----------------------------------------------------------------------===//
struct RockArithOpRewritePattern
    : public OpRewritePattern<rock::ArithOp> {
  using OpRewritePattern<rock::ArithOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::ArithOp op,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "RockArithOpRewritePattern: matchAndRewrite\n";
    
    Location loc = op.getLoc();
    
    // For now, just replace with tt.get_program_id for testing
    // TODO: Implement proper conversion based on op.getName()
    Value programId = GetProgramIdOp::create(rewriter, loc, 0);
    
    // Replace the arith_op result with the program_id
    rewriter.replaceOp(op, programId);
    
    return success();
  }
};

} // end anonymous namespace

void RockToTTIRPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  
  // Mark RockArithOp as illegal - it should be converted
  target.addIllegalOp<rock::ArithOp>();
  
  // Triton and Rock dialects are legal (Rock for now, will be converted later)
  target.addLegalDialect<triton::TritonDialect>();
  target.addLegalDialect<rock::RockDialect>();
  target.addLegalDialect<func::FuncDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<RockArithOpRewritePattern>(ctx);
  
  // Apply partial conversion - only convert RockArithOp, keep rest as-is
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}