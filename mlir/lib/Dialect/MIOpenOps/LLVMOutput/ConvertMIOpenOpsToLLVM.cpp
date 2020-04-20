//===- ConvertMIOpenOpsToLLVM.cpp - MLIR MIOpen ops lowering passes ---------------===//
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
// This pass converts miopen ops to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpenOps/ConvertMIOpenOpsToLLVM.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
struct LowerMIOpenOpsToLLVMPass : public FunctionPass<LowerMIOpenOpsToLLVMPass> {
  void runOnFunction() override;
};
} // end anonymous namespace

void LowerMIOpenOpsToLLVMPass::runOnFunction() {
  FuncOp func = getFunction();

  func.walk([&](miopen::TransformOp op) {
    op.replaceAllUsesWith(op.input());
    op.erase();
  });

  func.walk([&](miopen::FillOp op) {
    op.erase();
  });

  func.walk([&](miopen::LdsBarrierOp op) {
    op.erase();
  });

  func.walk([&](miopen::SubviewOp op) {
    op.replaceAllUsesWith(op.input());
    op.erase();
  });

  func.walk([&](miopen::ThreadwiseGemmOp op) {
    op.erase();
  });

  func.walk([&](miopen::ThreadwiseCopyOp op) {
    op.erase();
  });

  func.walk([&](miopen::GpuAllocOp op) {
    op.erase();
  });
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::miopen::createLowerMIOpenOpsToLLVMPass() {
  return std::make_unique<LowerMIOpenOpsToLLVMPass>();
}

static PassRegistration<LowerMIOpenOpsToLLVMPass>
    lowerMIOpenOpsToLLVMPass("miopen-lowering-llvm",
                       "Lower MIOpen ops to LLVM dialect.");
